import streamlit as st
import pickle
import re
import pandas as pd
import os

# ✅ Set page configuration first (before any other Streamlit command)
st.set_page_config(page_title="Spam Email Detector", page_icon="📩")

# Load the model and vectorizer
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, 'spam_detection_model.pkl')
vectorizer_path = os.path.join(BASE_DIR, 'tfidf_vectorizer.pkl')

model = pickle.load(open(model_path, 'rb'))
vectorizer = pickle.load(open(vectorizer_path, 'rb'))
# Function to clean and preprocess the input text
def preprocess_text(text):
    text = re.sub(r'\W', ' ', text)      # Remove non-word characters
    text = re.sub(r'\s+', ' ', text)     # Remove extra spaces
    text = text.lower()                  # Convert to lowercase
    return text

# Prediction function
def predict_spam(text):
    processed = preprocess_text(text)
    vectorized_text = vectorizer.transform([processed])
    prediction = model.predict(vectorized_text)
    return "🚫 Spam" if prediction[0] == 1 else "✅ Not Spam"

# --- Streamlit UI ---
st.title("📧 Spam Email Detection App")
st.write("Type or paste an email below to check if it's spam.")

# Text input
user_input = st.text_area("📨 Email content:")

# Predict button
if st.button("Check Email"):
    if user_input.strip() == "":
        st.warning("Please enter some text to check.")
    else:
        result = predict_spam(user_input)
        st.success(f"This email is classified as: **{result}**")

# Optional: Show mail_data.csv for reference
if st.checkbox("📂 Show sample dataset"):
    try:
        df = pd.read_csv("mail_data.csv")
        st.write(df.head(20))  # Show first 20 rows
    except FileNotFoundError:
        st.error("❌ `mail_data.csv` not found in the directory.")
