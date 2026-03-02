# Fake Job Posting Detection using NLP & Deep Learning

##  Overview

This project builds an end-to-end Natural Language Processing (NLP) pipeline to detect fraudulent job postings.

The system compares classical machine learning models and deep learning architectures (RNN, LSTM) while addressing severe class imbalance using SMOTE and class weighting techniques.

The final model is deployed via a Streamlit web application for real-time prediction.

---

##  Problem Statement

Online job platforms often contain fraudulent postings designed to scam applicants.

This project aims to:

- Classify job postings as **Real** or **Fake**
- Handle highly imbalanced dataset
- Compare ML vs DL performance
- Deploy a working inference application

---

##  Modeling Strategy

### 1. Text Preprocessing
- Combined multiple text fields (title, description, requirements, benefits, location)
- Removed special characters
- Lowercased text
- Removed English stopwords

---

### 2. Classical Machine Learning
Feature Engineering:
- TF-IDF (max_features = 5000)

Models:
- Naive Bayes
- Random Forest
- Support Vector Machine (SVM)

Imbalance Handling:
- SMOTE oversampling

Observation:
SMOTE significantly improved recall for minority (fraudulent) class.

---

### 3. Deep Learning Models
Sequence Processing:
- Tokenization (max_words = 10000)
- Padding (max_len = 300)

Architectures:
- SimpleRNN
- LSTM

Imbalance Handling:
- Class Weights
- SMOTE + Class Weights (best performance)

Final Best Model:
LSTM with SMOTE + Class Weight

Achieved strong balance between:
- Accuracy
- Precision
- Recall
- F1-Score

---

##  Deployment

A Streamlit application allows users to:

1. Input job description text
2. Run prediction using trained LSTM model
3. Receive classification result and confidence score

To run the app:

```bash
streamlit run app/streamlit_app.py


## Project Structure 

fake-job-detection-nlp/
│
├── train.py                 # Main training entry point
├── requirements.txt         # Project dependencies
│
├── src/                     # Core ML pipeline modules
│   ├── preprocess.py        # Data cleaning & preprocessing
│   ├── train_ml.py          # Classical ML models + SMOTE
│   ├── train_dl.py          # Deep Learning models
│
├── models/                  # Saved model artifacts (not tracked in Git)
│   ├── lstm_model.keras
│   └── tokenizer.joblib
│
├── app/                     # Deployment layer
│   └── streamlit_app.py
│
└── data/                    # Dataset (not included due to size)
    └── sample_dataset.csv
