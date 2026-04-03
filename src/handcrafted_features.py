import numpy as np
import re
import os
import pickle
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from textblob import TextBlob


BASE_DIR = os.path.dirname(os.path.dirname(__file__))

with open(os.path.join(BASE_DIR, "models", "label_top_words.pkl"), "rb") as f:
    label_top_words = pickle.load(f)

stop_words = ENGLISH_STOP_WORDS


def add_handcrafted_features(comment_text, preprocessed_text):

    comment_text = str(comment_text)
    preprocessed_text = str(preprocessed_text)

    words = preprocessed_text.split()
    words_set = set(words)

    sentence_count = len(re.findall(r'[.!?\n]+', comment_text))
    word_counts = len(words)
    unique_word_count = len(words_set)

    avg_word_len = np.mean([len(w) for w in words]) if words else 0

    uppercase_char_count = sum(1 for c in comment_text if c.isupper())
    uppercase_word_counts = sum(1 for w in comment_text.split() if w.isupper())

    hashtags_count = len([t for t in comment_text.split() if t.startswith('#')])
    mentions_count = len([t for t in comment_text.split() if t.startswith('@')])
    asterisk_count = len([t for t in comment_text.split() if t.startswith('*')])

    digits_count = sum(c.isdigit() for c in comment_text)
    stopword_count = sum(1 for w in comment_text.split() if w in stop_words)

    polarity = TextBlob(comment_text).sentiment.polarity
    subjectivity = TextBlob(comment_text).sentiment.subjectivity

    if polarity < 0:
        analysis = -1
    elif polarity > 0:
        analysis = 1
    else:
        analysis = 0

    toxic_words_count = sum(word in words_set for word in label_top_words.get('toxic', []))
    severe_toxic_count = sum(word in words_set for word in label_top_words.get('severe_toxic', []))
    obscene_words_count = sum(word in words_set for word in label_top_words.get('obscene', []))
    threat_words_count = sum(word in words_set for word in label_top_words.get('threat', []))
    insult_words_count = sum(word in words_set for word in label_top_words.get('insult', []))
    identity_hate_words_count = sum(word in words_set for word in label_top_words.get('identity_hate', []))

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
        subjectivity,
        polarity,
        analysis,
        toxic_words_count,
        severe_toxic_count,
        obscene_words_count,
        threat_words_count,
        insult_words_count,
        identity_hate_words_count
    ]

    return np.array(features).reshape(1, -1)