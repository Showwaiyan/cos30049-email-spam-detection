"""Feature extraction utilities for email spam detection."""

import string
import numpy as np
import pandas as pd
import emoji
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from .preprocess import clean_db_text, process_text_advanced

NUM_COLS = [
    "num_urls", "num_exclamation", "num_question", "num_dollar",
    "num_all_caps", "num_numbers", "word_count", "capital_ratio",
    "emoji_count"
]


def extract_features(email):
    """Extract numeric features and preprocessed text from a Series of raw emails."""
    db = pd.DataFrame()
    db["num_urls"]        = email.str.findall(r'https?://\S+|www\.\S+').str.len()
    db["num_exclamation"] = email.str.count(r'!')
    db["num_question"]    = email.str.count(r'\?')
    db["num_dollar"]      = email.str.count(r'\$')
    db["num_all_caps"]    = email.str.findall(r'\b[A-Z]{2,}\b').str.len()
    db["num_numbers"]     = email.str.findall(r'\d+').str.len()
    db["word_count"]      = email.str.split().str.len()
    caps   = email.str.findall(r'[A-Z]').str.len()
    letters = email.str.findall(r'[A-Za-z]').str.len()
    db["capital_ratio"]   = np.where(letters > 0, caps / letters, 0)
    db["emoji_count"]     = email.apply(emoji.emoji_count)
    db["text"] = email.apply(clean_db_text)
    db["text"] = db["text"].apply(process_text_advanced)
    return db

