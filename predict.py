'''
input text
→ preprocessing
→ feature engineering
→ TFIDF
→ handcrafted features
→ combine features
→ SVM model
→ predictions

import sys
sys.path.append("src")

import numpy as np
import pickle
import joblib
import re
import string

from scipy.sparse import hstack, csr_matrix
from preprocessing_text import text_cleaning


# labels
labels = ['toxic','severe_toxic','obscene','threat','insult','identity_hate']


# Load models
tfidf = joblib.load("models/tfidf_vectorizer.pkl")
scaler = joblib.load("models/handcrafted_scaler.pkl")

with open("models/final_svm_tfidf_handcrafted.pkl", "rb") as f:
    model = pickle.load(f)


def create_handcrafted_features(text):

    sentence_count = len(re.findall(r'[.!?\n]+', str(text)))

    words = text.split()

    word_counts = len(words)

    unique_word_count = len(set(words))

    avg_word_len = np.mean([len(w) for w in words]) if words else 0

    uppercase_char_count = sum(1 for c in text if c.isupper())

    uppercase_word_counts = sum(1 for w in words if w.isupper())

    hashtags_count = len([t for t in words if t.startswith('#')])

    mentions_count = len([t for t in words if t.startswith('@')])

    asterisk_count = len([t for t in words if t.startswith('*')])

    digits_count = sum(c.isdigit() for c in text)

    stopword_count = 0  # optional

    features = [
        sentence_count,
        word_counts,
        unique_word_count,
        avg_word_len,
        uppercase_char_count,
        uppercase_word_counts,
        hashtags_count,
        mentions_count,
        asterisk_count,
        digits_count,
        stopword_count,
        0,0,0, # sentiment placeholders
        0,0,0,0,0,0 # toxic word counts placeholders
    ]

    return np.array(features).reshape(1,-1)


def predict_comment(text):

    # preprocessing
    clean_text = text_cleaning(text)

    # tfidf
    tfidf_features = tfidf.transform([clean_text])

    # handcrafted
    handcrafted = create_handcrafted_features(text)

    # scale
    handcrafted_scaled = scaler.transform(handcrafted)

    handcrafted_sparse = csr_matrix(handcrafted_scaled)

    # combine
    final_features = hstack([tfidf_features, handcrafted_sparse])

    # predict
    prediction = model.predict(final_features)[0]

    result = dict(zip(labels, prediction))

    return result


if __name__ == "__main__":

    text = input("Enter comment: ")

    prediction = predict_comment(text)

    print("\nPrediction:")
    for label, value in prediction.items():
        print(f"{label}: {value}")
'''

import sys
sys.path.append("../src")

from preprocessing_text import text_cleaning

import pickle
import numpy as np
from scipy.sparse import hstack, csr_matrix
from joblib import load

# Labels
labels = ['toxic','severe_toxic','obscene','threat','insult','identity_hate']

# Load model
with open("models/final_svm_tfidf_handcrafted.pkl", "rb") as f:
    model = pickle.load(f)

# Load vectorizer
tfidf = load("models/tfidf_vectorizer.pkl")

# Dummy handcrafted feature generator
def handcrafted_features(text):

    word_count = len(text.split())
    char_count = len(text)
    uppercase = sum(1 for c in text if c.isupper())

    return np.array([[word_count, char_count, uppercase]])