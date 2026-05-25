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

app = Flask(__name__)

# Constants
CSV_PATH = "Language Detection.csv"
MAX_WORDS = 10000
MAX_LEN = 150
EMBEDDING_DIM = 128

# Download NLTK stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load spaCy model
try:
    nlp_en = spacy.load('en_core_web_sm')
except OSError:
    print("Run 'python -m spacy download en_core_web_sm'")
    sys.exit(1)

# Load and preprocess dataset
df = pd.read_csv(CSV_PATH)
df.dropna(inplace=True)
df['Text'] = df['Text'].astype(str)

label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['Language'])
LANGUAGES = list(label_encoder.classes_)
NUM_CLASSES = len(LANGUAGES)

def clean_text(text):
    return re.sub(r'[^\w\s]', '', text.lower().strip())

df['clean_text'] = df['Text'].apply(clean_text)

tokenizer = Tokenizer(num_words=MAX_WORDS)
tokenizer.fit_on_texts(df['clean_text'])
sequences = tokenizer.texts_to_sequences(df['clean_text'])
X = pad_sequences(sequences, maxlen=MAX_LEN)
y = tf.keras.utils.to_categorical(df['label'], NUM_CLASSES)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

def create_model():
    model = Sequential([
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
        Dense(NUM_CLASSES, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

model = create_model()
model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_val, y_val), verbose=1)

def preprocess_text(text):
    text = clean_text(text)
    if not text:
        return None, 0, 0, 0, 0, []

    doc = nlp_en(text)
    words = [token.text for token in doc if not token.is_punct]
    unique_words = len(set(words))
    char_count = len(text)
    special_chars = len(re.findall(r'[^a-zA-Z0-9\s]', text))
    keywords = [token.text for token in doc if not token.is_stop and not token.is_punct and len(token.text) > 1]

    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=MAX_LEN)

    return padded, len(words), char_count, unique_words, special_chars, keywords[:5]

def detect_language(text):
    padded, word_count, char_count, unique_words, special_chars, keywords = preprocess_text(text)
    if padded is None:
        return {'error': 'Empty text after preprocessing'}

    prediction = model.predict(padded, verbose=0)[0]
    idx = int(np.argmax(prediction))
    return {
        'language': LANGUAGES[idx],
        'confidence': float(prediction[idx]),
        'word_count': word_count,
        'char_count': char_count,
        'unique_words': unique_words,
        'special_chars': special_chars,
        'keywords': keywords,
        'probabilities': [float(p) for p in prediction]
    }

@app.route('/')
def index():
    return render_template_string(open("index.html", encoding="utf-8").read())

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

