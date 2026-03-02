"""
train.py

Main training entry point for Fake Job Detection project.
"""

from src.preprocess import load_and_preprocess_data
from src.train_dl import prepare_sequences, train_smote_classweight

DATA_PATH = "data/sample_dataset.csv"

def main():
    print("Loading and preprocessing data...")
    df = load_and_preprocess_data(DATA_PATH)

    print("Preparing sequences...")
    X, y, tokenizer = prepare_sequences(df)

    print("Training LSTM with SMOTE + Class Weight...")
    train_smote_classweight(X, y, tokenizer)

    print("Training complete. Model saved in /models directory.")

if __name__ == "__main__":
    main()
