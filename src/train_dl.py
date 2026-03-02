"""
train_dl.py

This module handles Deep Learning training for the Fake Job Detection project.

Workflow:
1. Tokenize and pad sequences
2. Train RNN and LSTM models (baseline)
3. Apply class weights
4. Apply SMOTE + class weights
5. Save trained LSTM model
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


-
# Step 1: Tokenization & Padding
def prepare_sequences(df, max_words=10000, max_len=300):
    """
    Convert cleaned text into padded sequences.
    """

    tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
    tokenizer.fit_on_texts(df["cleaned_text"])

    sequences = tokenizer.texts_to_sequences(df["cleaned_text"])
    padded = pad_sequences(sequences, maxlen=max_len, padding="post")

    y = df["fraudulent"]

    return padded, y, tokenizer


-
# Step 2: Build Models 
def build_rnn_model(max_words=10000, max_len=300):
    model = Sequential([
        Embedding(input_dim=max_words, output_dim=64, input_length=max_len),
        SimpleRNN(64),
        Dense(1, activation="sigmoid")
    ])

    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model


def build_lstm_model(max_words=10000, max_len=300):
    model = Sequential([
        Embedding(input_dim=max_words, output_dim=64, input_length=max_len),
        LSTM(64),
        Dense(1, activation="sigmoid")
    ])

    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model



# Step 3: Train Baseline DL Models
def train_baseline_dl(X, y, epochs=5, batch_size=64):

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    rnn_model = build_rnn_model()
    rnn_model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2)

    lstm_model = build_lstm_model()
    lstm_model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2)

    return rnn_model, lstm_model



# Step 4: Train with Class Weights-
def train_with_class_weights(X, y, epochs=5, batch_size=64):

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    class_weights = dict(zip(classes, weights))

    lstm_model = build_lstm_model()
    lstm_model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        class_weight=class_weights,
    )

    return lstm_model



# Step 5: SMOTE + Class Weight 
import os
import joblib

def train_smote_classweight(
    X, y, tokenizer,
    epochs=5,
    batch_size=64,
    model_path="models/lstm_model.keras",
    tokenizer_path="models/tokenizer.joblib"
):
    """
    Train LSTM model using SMOTE + class weights.
    Save both model and tokenizer.
    """

    # Ensure models directory exists
    os.makedirs("models", exist_ok=True)

    # Step 1: Apply SMOTE
    smote = SMOTE(random_state=42)
    X_smote, y_smote = smote.fit_resample(X, y)

    # Step 2: Compute class weights
    classes = np.unique(y_smote)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_smote)
    class_weights = dict(zip(classes, weights))

    # Step 3: Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_smote, y_smote, test_size=0.2, random_state=42
    )

    # Step 4: Build & train LSTM
    lstm_model = build_lstm_model()
    lstm_model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        class_weight=class_weights,
    )

    # Step 5: Save model and tokenizer
    lstm_model.save(model_path)
    joblib.dump(tokenizer, tokenizer_path)

    return lstm_model
