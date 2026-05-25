from flask import Flask, render_template_string, request, jsonify
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dense, Flatten, Embedding, Dropout, LSTM
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import spacy
import nltk
from nltk.corpus import stopwords
import re
import sys
import os
import pickle
import subprocess

app = Flask(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
CSV_PATH       = os.path.join(BASE_DIR, "Language Detection.csv")
MODEL_PATH     = os.path.join(BASE_DIR, "lang_model.keras")
TOKENIZER_PATH = os.path.join(BASE_DIR, "tokenizer.pkl")
ENCODER_PATH   = os.path.join(BASE_DIR, "label_encoder.pkl")
MAX_WORDS      = 10000
MAX_LEN        = 150
EMBEDDING_DIM  = 128

# ── NLTK stopwords ────────────────────────────────────────────────────────────
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

# ── spaCy (auto-download if missing) ─────────────────────────────────────────
try:
    nlp_en = spacy.load('en_core_web_sm')
except OSError:
    print("⬇ Downloading spaCy en_core_web_sm ...")
    subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"], check=True)
    nlp_en = spacy.load('en_core_web_sm')

# ── Helpers ───────────────────────────────────────────────────────────────────
def clean_text(text):
    return re.sub(r'[^\w\s]', '', text.lower().strip())

def create_model(num_classes):
    m = Sequential([
        Embedding(MAX_WORDS, EMBEDDING_DIM, input_length=MAX_LEN),
        Conv1D(128, 5, activation='relu', padding='same'),
        MaxPooling1D(2),
        LSTM(64, return_sequences=True),
        Dropout(0.3),
        Conv1D(64, 3, activation='relu', padding='same'),
        MaxPooling1D(2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    m.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return m

# ── Load saved model OR train from scratch ────────────────────────────────────
if os.path.exists(MODEL_PATH) and os.path.exists(TOKENIZER_PATH) and os.path.exists(ENCODER_PATH):
    print("✅ Loading saved model ...")
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(TOKENIZER_PATH, 'rb') as f:
        tokenizer = pickle.load(f)
    with open(ENCODER_PATH, 'rb') as f:
        label_encoder = pickle.load(f)
    LANGUAGES   = list(label_encoder.classes_)
    NUM_CLASSES = len(LANGUAGES)

else:
    print("🔄 Training model from scratch ...")
    df = pd.read_csv(CSV_PATH)
    df.dropna(inplace=True)
    df['Text'] = df['Text'].astype(str)

    label_encoder = LabelEncoder()
    df['label']   = label_encoder.fit_transform(df['Language'])
    LANGUAGES     = list(label_encoder.classes_)
    NUM_CLASSES   = len(LANGUAGES)

    df['clean_text'] = df['Text'].apply(clean_text)

    tokenizer = Tokenizer(num_words=MAX_WORDS)
    tokenizer.fit_on_texts(df['clean_text'])

    sequences = tokenizer.texts_to_sequences(df['clean_text'])
    X = pad_sequences(sequences, maxlen=MAX_LEN)
    y = tf.keras.utils.to_categorical(df['label'], NUM_CLASSES)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    model = create_model(NUM_CLASSES)
    model.fit(X_train, y_train, epochs=5, batch_size=32,
              validation_data=(X_val, y_val), verbose=1)

    # ── Save so Render won't retrain on every restart ──
    model.save(MODEL_PATH)
    with open(TOKENIZER_PATH, 'wb') as f:
        pickle.dump(tokenizer, f)
    with open(ENCODER_PATH, 'wb') as f:
        pickle.dump(label_encoder, f)
    print("✅ Model saved.")

# ── Inference ─────────────────────────────────────────────────────────────────
def preprocess_text(text):
    text = clean_text(text)
    if not text:
        return None, 0, 0, 0, 0, []
    doc          = nlp_en(text)
    words        = [t.text for t in doc if not t.is_punct]
    unique_words = len(set(words))
    char_count   = len(text)
    special_chars= len(re.findall(r'[^a-zA-Z0-9\s]', text))
    keywords     = [t.text for t in doc if not t.is_stop and not t.is_punct and len(t.text) > 1]
    seq    = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=MAX_LEN)
    return padded, len(words), char_count, unique_words, special_chars, keywords[:5]

def detect_language(text):
    padded, word_count, char_count, unique_words, special_chars, keywords = preprocess_text(text)
    if padded is None:
        return {'error': 'Empty text after preprocessing'}
    prediction = model.predict(padded, verbose=0)[0]
    idx = int(np.argmax(prediction))
    return {
        'language'     : LANGUAGES[idx],
        'confidence'   : float(prediction[idx]),
        'word_count'   : word_count,
        'char_count'   : char_count,
        'unique_words' : unique_words,
        'special_chars': special_chars,
        'keywords'     : keywords,
        'probabilities': [float(p) for p in prediction],
        'all_languages': LANGUAGES,
    }

# ── Routes ────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    html_path = os.path.join(BASE_DIR, 'index.html')
    with open(html_path, encoding='utf-8') as f:
        return render_template_string(f.read())

@app.route('/detect', methods=['POST'])
def detect():
    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        result = detect_language(text)
        if 'error' in result:
            return jsonify(result), 400
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
