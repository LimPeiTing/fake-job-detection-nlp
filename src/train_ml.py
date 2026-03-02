"""
train_ml.py

This module handles classical machine learning training
for the Fake Job Detection project.

Workflow:
1. Convert cleaned text into TF-IDF features
2. Split data into train/test sets
3. Train ML models (Naive Bayes, Random Forest, SVM)
4. Apply SMOTE for class imbalance
5. Evaluate model performance
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from imblearn.over_sampling import SMOTE



# Step 1: Convert text into TF-IDF features
def generate_tfidf_features(df, max_features=5000):
    """
    Convert cleaned text into TF-IDF representation.
    """
    tfidf_vectorizer = TfidfVectorizer(max_features=max_features)
    X_tfidf = tfidf_vectorizer.fit_transform(df["cleaned_text"])
    y = df["fraudulent"]

    return X_tfidf, y, tfidf_vectorizer



# Step 2: Train/Test Split
def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)



# Step 3: Evaluation Function 
def evaluate_model(y_true, y_pred, model_name):
    print(f"\n--- {model_name} ---")
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred))
    print("Recall:", recall_score(y_true, y_pred))
    print("F1 Score:", f1_score(y_true, y_pred))
    print("ROC AUC:", roc_auc_score(y_true, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))



# Step 4: Train ML Models (Without SMOTE)
def train_ml_models(X_train, X_test, y_train, y_test):
    models = {
        "Naive Bayes": MultinomialNB(),
        "Random Forest": RandomForestClassifier(random_state=42),
        "SVM": LinearSVC(),
    }

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        evaluate_model(y_test, y_pred, name)



# Step 5: Train ML Models with SMOTE
def train_ml_with_smote(X_train, X_test, y_train, y_test):
    smote = SMOTE(random_state=42)
    X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

    models = {
        "Naive Bayes (SMOTE)": MultinomialNB(),
        "Random Forest (SMOTE)": RandomForestClassifier(random_state=42),
        "SVM (SMOTE)": LinearSVC(),
    }

    for name, model in models.items():
        model.fit(X_train_sm, y_train_sm)
        y_pred = model.predict(X_test)
        evaluate_model(y_test, y_pred, name)
