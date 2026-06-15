# Step 1: Import Libraries and Load the Model
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model

# Load the IMDB dataset word index
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

# Load the pre-trained model with ReLU activation
model = load_model('simple_rnn_imdb.h5')

# Step 2: Helper Functions
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review


import streamlit as st

# Page setup
st.set_page_config(page_title="Movie Sentiment AI", page_icon="🎬", layout="centered")

# Custom CSS for attractive look
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    }
    h1 {
        color: #ffd700;
        text-align: center;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
    }
    .stTextArea textarea {
        background-color: #2d2d44;
        color: #ffffff;
        border: 2px solid #ffd700;
        border-radius: 10px;
    }
    .stButton button {
        background-color: #ffd700;
        color: #1e1e2f;
        font-weight: bold;
        border-radius: 25px;
        padding: 10px 30px;
        border: none;
        transition: 0.3s;
        display: block;
        margin: 0 auto;
    }
    .stButton button:hover {
        background-color: #ffec80;
        transform: scale(1.05);
    }
    .result-box {
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        margin-top: 20px;
    }
    .positive {
        background-color: rgba(46, 204, 113, 0.2);
        border: 2px solid #2ecc71;
        color: #2ecc71;
    }
    .negative {
        background-color: rgba(231, 76, 60, 0.2);
        border: 2px solid #e74c3c;
        color: #e74c3c;
    }
</style>
""", unsafe_allow_html=True)

# Streamlit app
st.title('🎬 IMDB Movie Review Sentiment Analysis')
st.write('Enter a movie review to classify it as positive or negative.')

# User input
user_input = st.text_area('Movie Review', placeholder="Type or paste your review here...")

if st.button('Classify'):

    preprocessed_input = preprocess_text(user_input)

    ## Make prediction
    prediction = model.predict(preprocessed_input)
    score = prediction[0][0]
    sentiment = 'Positive' if score > 0.5 else 'Negative'

    emoji = "😃" if sentiment == "Positive" else "😞"
    css_class = "positive" if sentiment == "Positive" else "negative"

    # Display the result
    st.markdown(f"""
    <div class="result-box {css_class}">
        {emoji} {sentiment}
    </div>
    """, unsafe_allow_html=True)

    st.progress(float(score))
    st.write(f'Prediction Score: {score}')
else:
    st.write('Please enter a movie review.')