import streamlit as st
import pickle
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from pathlib import Path


# Function to load models and vectorizers
def load_model():
    # Load the model and vectorizer
    model = pickle.load(open("Models/model.pkl", "rb"))  # Use forward slashes for paths
    vectorizer = pickle.load(open("Models/vectorizer.pkl", "rb"))
    return model, vectorizer


# Load model and vectorizer
model, vectorizer = load_model()


# Function to predict sentiment
def predict_sentiment(text):
    text_vectorized = vectorizer.transform([text])
    prediction = model.predict(text_vectorized)
    return prediction[0]


# Set up Streamlit interface
st.set_page_config(
    page_title="Sentiment Analysis", page_icon=":chart_with_upwards_trend:"
)

# Add custom styling for better appearance
st.markdown(
    """
    <style>
    .main { 
        background-color: #f0f2f6;
    }
    h1 {
        color: #4CAF50;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
    }
    .stTextInput>div>input {
        background-color: #ffffff;
        color: #333;
        border: 2px solid #4CAF50;
        padding: 10px;
    }
    .stTextInput>div>input:focus {
        border-color: #66bb6a;
    }
    </style>
""",
    unsafe_allow_html=True,
)

# Display app header
st.title("Sentiment Analysis App")
st.markdown(
    "This app predicts the sentiment of a given text, identifying if it's positive or negative."
)

# Add input field for user text
text_input = st.text_area("Enter the text for sentiment analysis:")

# Prediction button
if st.button("Analyze Sentiment"):
    if text_input:
        sentiment = predict_sentiment(text_input)
        if sentiment == 1:
            st.success("The sentiment is **Positive** 😊")
        else:
            st.error("The sentiment is **Negative** 😞")
    else:
        st.warning("Please enter some text to analyze.")

# Footer
st.markdown(
    """
    <div style='text-align: center; color: #555555; font-size: 12px;'>
        <p>Created with Sayed Abdalsamie</p>
    </div>
""",
    unsafe_allow_html=True,
)
