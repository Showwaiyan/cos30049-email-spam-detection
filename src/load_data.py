"""Data loading utilities for email spam datasets."""

import pandas as pd
from .config import DATA_PATHS


def load_enron_spam(encoding="utf-8"):
    """Load the Enron spam dataset.

    Returns:
        DataFrame with columns: 'text', 'label'
    """
    df = pd.read_csv(DATA_PATHS["enron"], encoding=encoding)
    df = df.drop(columns=["Message ID", "Date"], errors="ignore")
    df = df.dropna(subset=["Message"])
    df["Subject"] = df["Subject"].fillna("")

    df["Spam/Ham"] = df["Spam/Ham"].astype(str).str.strip().str.lower()
    df["label"] = df["Spam/Ham"].map({"ham": 0, "spam": 1})
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)

    df["text"] = df["Subject"].astype(str) + " " + df["Message"].astype(str)

    return df[["text", "label"]]


def load_enron_spam_2(encoding="utf-8"):
    """Load the Enron spam dataset (version 2).

    Returns:
        DataFrame with columns: 'text', 'label'
    """
    df = pd.read_csv(DATA_PATHS["enron_2"], encoding=encoding)
    df = df.dropna(subset=["body"])
    df["subject"] = df["subject"].fillna("")

    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df["label"] = df["label"].map({"ham": 0, "spam": 1, "0": 0, "1": 1})
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)

    df["text"] = df["subject"].astype(str) + " " + df["body"].astype(str)

    return df[["text", "label"]]


def load_spam_assassin(encoding="utf-8"):
    """Load the SpamAssassin dataset.

    Returns:
        DataFrame with columns: 'text', 'label'
    """
    df = pd.read_csv(DATA_PATHS["spam_assassin"], encoding=encoding)
    df = df.rename(columns={"target": "label"})
    df["label"] = df["label"].map({0: 0, 1: 1})
    df = df.dropna(subset=["text", "label"])
    df["label"] = df["label"].astype(int)

    return df[["text", "label"]]


def load_spam_assassin_2(encoding="utf-8"):
    """Load the SpamAssassin dataset (version 2).

    Returns:
        DataFrame with columns: 'text', 'label'
    """
    df = pd.read_csv(DATA_PATHS["spam_assassin_2"], encoding=encoding)
    df = df.dropna(subset=["body"])
    df["subject"] = df["subject"].fillna("")

    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df["label"] = df["label"].map({"ham": 0, "spam": 1, "0": 0, "1": 1})
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)

    df["text"] = df["subject"].astype(str) + " " + df["body"].astype(str)

    return df[["text", "label"]]


def load_email_spam_dataset(encoding="utf-8"):
    """Load the email spam dataset.

    Returns:
        DataFrame with columns: 'text', 'label'
    """
    df = pd.read_csv(DATA_PATHS["email_spam_dataset"], encoding=encoding)
    df = df.rename(columns={"email_text": "text"})

    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df["label"] = df["label"].map({"ham": 0, "spam": 1, "0": 0, "1": 1})
    df = df.dropna(subset=["text", "label"])
    df["label"] = df["label"].astype(int)

    return df[["text", "label"]]


def load_phishing_email(encoding="utf-8"):
    """Load the phishing email dataset.

    Returns:
        DataFrame with columns: 'text', 'label'
    """
    df = pd.read_csv(DATA_PATHS["phishing_email"], encoding=encoding)
    df = df.rename(columns={"text_combined": "text"})

    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df["label"] = df["label"].map({"ham": 0, "spam": 1, "0": 0, "1": 1})
    df = df.dropna(subset=["text", "label"])
    df["label"] = df["label"].astype(int)

    return df[["text", "label"]]


def load_sms_spam_dataset(encoding="utf-8"):
    """Load the SMS spam dataset.

    Returns:
        DataFrame with columns: 'text', 'label'
    """
    df = pd.read_csv(DATA_PATHS["sms_spam_dataset"], encoding=encoding)
    df = df.rename(columns={"v1": "label", "v2": "text"})

    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df["label"] = df["label"].map({"ham": 0, "spam": 1})
    df = df.dropna(subset=["text", "label"])
    df["label"] = df["label"].astype(int)

    return df[["text", "label"]]


def load_spam_email_detection_dataset(encoding="utf-8"):
    """Load the spam email detection dataset.

    Returns:
        DataFrame with columns: 'text', 'label'
    """
    df = pd.read_csv(DATA_PATHS["spam_email_detection_dataset"], encoding=encoding)
    df = df.dropna(subset=["email_text"])
    df["subject"] = df["subject"].fillna("")

    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df["label"] = df["label"].map({"ham": 0, "spam": 1, "0": 0, "1": 1})
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)

    df["text"] = df["subject"].astype(str) + " " + df["email_text"].astype(str)

    return df[["text", "label"]]


def load_250K_email_dataset(encoding="utf-8"):
    """Load the 250K+ email dataset.

    Returns:
        DataFrame with columns: 'text', 'label'
    """
    df = pd.read_csv(DATA_PATHS["250K_email_dataset"], encoding=encoding)

    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df["label"] = df["label"].map({"ham": 0, "spam": 1, "0": 0, "2": 1})
    df = df.dropna(subset=["text", "label"])
    df["label"] = df["label"].astype(int)

    return df[["text", "label"]]


def load_190K_email_content(encoding="utf-8"):
    """Load the 190K+ email content dataset.

    Returns:
        DataFrame with columns: 'text', 'label'
    """
    df = pd.read_csv(DATA_PATHS["190K_email_content"], encoding=encoding)

    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df["label"] = df["label"].map({"ham": 0, "spam": 1, "0": 0, "1": 1})
    df = df.dropna(subset=["text", "label"])
    df["label"] = df["label"].astype(int)

    return df[["text", "label"]]


def load_NLP_spam_ham_email(encoding="utf-8"):
    """Load the NLP spam ham email dataset.

    Returns:
        DataFrame with columns: 'text', 'label'
    """
    df = pd.read_csv(DATA_PATHS["NLP_spam_ham_email"], encoding=encoding)
    df = df.rename(columns={"spam": "label"})

    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df["label"] = df["label"].map({"ham": 0, "spam": 1, "0": 0, "1": 1})
    df = df.dropna(subset=["text", "label"])
    df["label"] = df["label"].astype(int)

    return df[["text", "label"]]


def load_ling_spam(encoding="utf-8"):
    """Load the ling spam dataset.

    Returns:
        DataFrame with columns: 'text', 'label'
    """
    df = pd.read_csv(DATA_PATHS["ling_spam"], encoding=encoding)
    df = df.dropna(subset=["message"])
    df["subject"] = df["subject"].fillna("")

    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df["label"] = df["label"].map({"ham": 0, "spam": 1, "0": 0, "1": 1})
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)

    df["text"] = df["subject"].astype(str) + " " + df["message"].astype(str)

    return df[["text", "label"]]


def load_nigerian_fraud(encoding="utf-8"):
    """Load the Nigerian fraud dataset.

    Returns:
        DataFrame with columns: 'text', 'label'
    """
    df = pd.read_csv(DATA_PATHS["nigerian_fraud"], encoding=encoding)
    df = df.dropna(subset=["body"])
    df["subject"] = df["subject"].fillna("")

    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df["label"] = df["label"].map({"ham": 0, "spam": 1, "0": 0, "1": 1})
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)

    df["text"] = df["subject"].astype(str) + " " + df["body"].astype(str)

    return df[["text", "label"]]


def load_nazario(encoding="utf-8"):
    """Load the Nazario dataset.

    Returns:
        DataFrame with columns: 'text', 'label'
    """
    df = pd.read_csv(DATA_PATHS["nazario"], encoding=encoding)
    df = df.dropna(subset=["body"])
    df["subject"] = df["subject"].fillna("")

    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df["label"] = df["label"].map({"ham": 0, "spam": 1, "0": 0, "1": 1})
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)

    df["text"] = df["subject"].astype(str) + " " + df["body"].astype(str)

    return df[["text", "label"]]


def load_CEAS_08(encoding="utf-8"):
    """Load the CEAS 08 dataset.

    Returns:
        DataFrame with columns: 'text', 'label'
    """
    df = pd.read_csv(DATA_PATHS["CEAS_08"], encoding=encoding)
    df = df.dropna(subset=["body"])
    df["subject"] = df["subject"].fillna("")

    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df["label"] = df["label"].map({"ham": 0, "spam": 1, "0": 0, "1": 1})
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)

    df["text"] = df["subject"].astype(str) + " " + df["body"].astype(str)

    return df[["text", "label"]]
