"""
streamlit_app.py

Streamlit deployment app for Fake Job Detection (LSTM model).

This app:
1. Loads trained LSTM model
2. Loads tokenizer
3. Preprocesses user input
4. Predicts whether job posting is Real or Fake
"""

import streamlit as st
import tensorflow as tf
import joblib
from tensorflow.keras.preprocessing.sequence import pad_sequences


# Configuration
MODEL_PATH = "models/lstm_model.keras"
TOKENIZER_PATH = "models/tokenizer.joblib"
MAX_LEN = 300  # match with training configuration



# Load Model & Tokenizer
@st.cache_resource
def load_model_and_tokenizer():
    model = tf.keras.models.load_model(MODEL_PATH)
    tokenizer = joblib.load(TOKENIZER_PATH)
    return model, tokenizer


model, tokenizer = load_model_and_tokenizer()



# Prediction Function
def predict_fake_job(text):
    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=MAX_LEN, padding="post", truncating="post")

    prediction = model.predict(padded)[0][0]

    label = "Fake" if prediction >= 0.5 else "Real"
    confidence = round(prediction * 100, 2) if label == "Fake" else round((1 - prediction) * 100, 2)

    return label, confidence



# Streamlit UI
st.title("Fake Job Posting Detector (LSTM)")
st.write("Enter a job description to classify whether it is real or fraudulent.")

job_text = st.text_area("Job Description", "")

if st.button("Predict"):
    if job_text.strip():
        label, confidence = predict_fake_job(job_text)
        st.success(f"Prediction: **{label}**")
        st.info(f"Confidence: {confidence}%")
    else:
        st.warning("Please enter job description text.")
