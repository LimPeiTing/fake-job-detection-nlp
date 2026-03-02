"""
preprocess.py

This module handles all data loading and text preprocessing steps
for the Fake Job Detection NLP project.

The preprocessing workflow follows these stages:

1. Load dataset
2. Remove irrelevant columns
3. Handle missing values
4. Combine text-related columns
5. Clean raw text (remove symbols, lowercase, remove stopwords)
"""

import pandas as pd
import re
import nltk
from nltk.corpus import stopwords


# Download stopwords
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))


# Step 1: Text Cleaning Function
def clean_text(text: str) -> str:
    """
    Clean raw text by:
    - Removing non-alphabet characters
    - Converting to lowercase
    - Removing English stopwords

    Parameters:
        text (str): Raw input text

    Returns:
        str: Cleaned text
    """

    # Remove non-alphabet characters
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    # Convert to lowercase
    text = text.lower()

    # Tokenize text
    tokens = text.split()

    # Remove stopwords
    tokens = [word for word in tokens if word not in stop_words]

    return " ".join(tokens)


# Step 2: Load and Preprocess Dataset
def load_and_preprocess_data(file_path: str) -> pd.DataFrame:
    """
    Load dataset and perform preprocessing steps.

    Workflow:
    1. Load CSV file
    2. Drop irrelevant columns
    3. Handle missing values
    4. Combine text-related columns into single field
    5. Apply text cleaning

    Parameters:
        file_path (str): Path to dataset CSV file

    Returns:
        pd.DataFrame: Preprocessed DataFrame
    """

    # Step 2.1: Load dataset
    df = pd.read_csv(file_path)

    # Step 2.2: Drop irrelevant column
    if "job_id" in df.columns:
        df = df.drop(columns=["job_id"])

    # Step 2.3: Replace missing values with empty string
    df.fillna("", inplace=True)

    # Step 2.4: Combine important text columns
    text_columns = [
        "title",
        "company_profile",
        "description",
        "requirements",
        "benefits",
        "location",
    ]

    df["text"] = df[text_columns].agg(" ".join, axis=1)

    # Step 2.5: Clean combined text
    df["cleaned_text"] = df["text"].apply(clean_text)

    return df
