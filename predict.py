'''
input text
→ preprocessing
→ feature engineering
→ TFIDF
→ handcrafted features
→ combine features
→ SVM model
→ predictions
'''

import sys
sys.path.append("src")

from preprocessing_text import text_cleaning
from handcrafted_features import add_handcrafted_features

import pickle
import numpy as np
from scipy.sparse import hstack, csr_matrix
from joblib import load

# Load model
with open("models/final_svm_tfidf_handcrafted.pkl", "rb") as f:
    model = pickle.load(f)

# Load vectorizer
tfidf = load("models/tfidf_vectorizer.pkl")

# Labels
labels = ['toxic','severe_toxic','obscene','threat','insult','identity_hate']

def predict_comment(text: str):
    """Run the full prediction pipeline for one comment."""

    if not text.strip():
        raise ValueError("Input text is empty.")

    # Preprocessing the text
    preprocessed_text = text_cleaning(text)

    # Creating features
    handcrafted_features = add_handcrafted_features(text, preprocessed_text)

    # Creating tfidf features
    tfidf_features = tfidf.transform([preprocessed_text])

    # Convert handcrafted to sparse
    handcrafted_sparse = csr_matrix(handcrafted_features)

    # Combine features
    features_set = hstack([tfidf_features, handcrafted_sparse])

    # model prediction
    prediction = model.predict(features_set)[0]

    return dict(zip(labels, prediction))


def main():
    text = input("Enter the comment: ")

    try:
        result = predict_comment(text)

        print("\nPrediction:")
        for label, value in result.items():
            print(f"{label}: {value}")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()