import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
from flask import Flask, request, jsonify, render_template_string

# Flask app
app = Flask(__name__)

# --- Load Dataset and Prepare Tokenizer ---
def load_dataset_and_prepare_tokenizer(dataset_path, max_sequence_length=100):
    """Load dataset containing multilingual words and prepare tokenizer."""
    file_path = r"C:\Language Detection.csv"  # Correct file path

    # Load dataset with specified encoding
    data = pd.read_csv(file_path, encoding='ISO-8859-1')  # Try 'ISO-8859-1', 'latin1', or 'windows-1252'
    print(data.head())

    # Extract texts and labels
    texts = data['text'].tolist()
    labels = data['label'].tolist()

    # Initialize and fit the tokenizer
    tokenizer = Tokenizer(char_level=True, oov_token='<UNK>')
    tokenizer.fit_on_texts(texts)

    # Preprocess texts into padded sequences
    text_sequences = tokenizer.texts_to_sequences(texts)
    padded_texts = pad_sequences(text_sequences, maxlen=max_sequence_length, padding='post', truncating='post')

    return tokenizer, padded_texts, labels

# Load the dataset and tokenizer
dataset_path = '/mnt/data/Language Detection.csv'  # Path to the uploaded dataset
max_sequence_length = 100
tokenizer, _, _ = load_dataset_and_prepare_tokenizer(dataset_path, max_sequence_length)

# Load pre-trained model
model_path = 'language_detection_model.h5'
model = tf.keras.models.load_model(model_path)

# Define language map
language_map = {
    0: 'English',
    1: 'French',
    2: 'Spanish',
    3: 'German',
    4: 'Italian',
    5: 'Portuguese',
    6: 'Swedish',
    7: 'Dutch',
    # Add more languages if needed
}

# HTML Template
html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Language Detection</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            text-align: center;
            margin: 50px;
        }
        .container {
            max-width: 400px;
            margin: auto;
            padding: 20px;
            border: 1px solid #ddd;
            border-radius: 10px;
            box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1);
        }
        input[type="text"], button {
            width: 100%;
            padding: 10px;
            margin: 10px 0;
            border: 1px solid #ccc;
            border-radius: 5px;
            box-sizing: border-box;
        }
        button {
            background-color: #007BFF;
            color: white;
            border: none;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Language Detection</h1>
        <form id="languageForm">
            <input type="text" id="text" name="text" placeholder="Enter a sentence..." required>
            <button type="submit">Detect Language</button>
        </form>
        <p id="result"></p>
    </div>

    <script>
        document.getElementById('languageForm').addEventListener('submit', function(event) {
            event.preventDefault();
            const text = document.getElementById('text').value;

            fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ text: text }),
            })
            .then(response => response.json())
            .then(data => {
                document.getElementById('result').innerText = `Predicted Language: ${data.language}`;
            })
            .catch(error => {
                console.error('Error:', error);
                document.getElementById('result').innerText = 'Error detecting language.';
            });
        });
    </script>
</body>
</html>
"""

# Route for home page
@app.route('/')
def home():
    return render_template_string(html_template)

# Route for language prediction
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    input_text = data.get('text', '')

    # Preprocess input text
    input_sequence = tokenizer.texts_to_sequences([input_text])
    padded_text = pad_sequences(input_sequence, maxlen=max_sequence_length, padding='post', truncating='post')

    # Predict the language
    predictions = model.predict(padded_text)
    predicted_language_index = np.argmax(predictions, axis=1)[0]
    predicted_language = language_map.get(predicted_language_index, "Unknown Language")

    return jsonify({'language': predicted_language})

if __name__ == '__main__':
    app.run(debug=True)