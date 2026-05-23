"""Text preprocessing utilities for email spam detection."""

import re
from html.parser import HTMLParser
import emoji

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Run once to download required NLTK data
nltk.download("stopwords")
nltk.download("wordnet")

class TextPreprocessor:
    def __init__(self, text=""):
        self.text = text

    def strip_html_tags(self):
        self.text = re.sub(r"<[^>]+>", "", self.text)
        return self

    def replace_phone_numbers(self):
        phone_re = r"\+?[\d]{1,3}[-.\s]?\(?\d{1,4}\)?[-.\s]?\d{1,4}[-.\s]?\d{1,9}"
        self.text = re.sub(phone_re, " [PhoneNumber] ", self.text)
        return self

    def replace_urls(self):
        self.text = re.sub(r"http[s]?://\S+", " [URL] ", self.text)
        return self

    def normalize_whitespace(self):
        self.text = re.sub(r"\s+", " ", self.text).strip()
        return self

    def to_lowercase(self):
        self.text = self.text.lower()
        return self

    def remove_special_characters(self):
        self.text = re.sub(r"[:?,.*]", "", self.text)
        return self

    def replace_emojis(self):
        self.text = re.sub(
            r":([a-z0-9_]+):",
            lambda m: "[" + "".join(w.capitalize() for w in m.group(1).split("_")) + "EMOJI]",
            emoji.demojize(self.text),
        )
        return self

    def replace_emails(self):
        email_re = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
        self.text = re.sub(email_re, " [EMAIL] ", self.text)
        return self

    def replace_percentages(self):
        self.text = re.sub(r"\b\d+(?:\.\d+)?\s?%", " [PERCENTAGE] ", self.text)
        return self

    def replace_numbers(self):
        self.text = re.sub(r"\b\d+\b", " [NUMBER] ", self.text)
        return self

    def remove_stopwords(self):
        stop_words = set(stopwords.words("english"))
        # Preserve custom tokens like [URL], [EMAIL], etc.
        tokens = self.text.split()
        self.text = " ".join(w for w in tokens if w.lower() not in stop_words or w.startswith("["))
        return self

    def lemmatize(self):
        lemmatizer = WordNetLemmatizer()
        tokens = self.text.split()
        self.text = " ".join(lemmatizer.lemmatize(w) for w in tokens)
        return self

    def get_text(self):
        return self.text


class HTMLTextExtractor(HTMLParser):
    """Strip HTML tags, skip style/script blocks, extract visible text only."""

    def __init__(self):
        super().__init__()
        self.text_parts = []
        self._skip = False

    def handle_starttag(self, tag, attrs):
        if tag.lower() in ("style", "script"):
            self._skip = True

    def handle_endtag(self, tag):
        if tag.lower() in ("style", "script"):
            self._skip = False

    def handle_data(self, data):
        if self._skip:
            return
        stripped = data.strip()
        if stripped:
            self.text_parts.append(stripped)

    def get_text(self):
        return " ".join(self.text_parts)


def parse_eml(filepath):
    """Parse .eml email file and extract subject and body.

    Args:
        filepath: Path to .eml file

    Returns:
        tuple: (subject, body)
    """
    import email

    with open(filepath, "rb") as f:
        msg = email.message_from_bytes(f.read())

    subject = msg.get("Subject", "")

    plain_body = ""
    html_body = ""

    if msg.is_multipart():
        for part in msg.walk():
            content_type = part.get_content_type()
            payload = part.get_payload(decode=True)
            if not payload:
                continue
            decoded = payload.decode("utf-8", errors="ignore")
            if content_type == "text/plain" and not plain_body:
                plain_body = decoded
            elif content_type == "text/html" and not html_body:
                html_body = decoded
    else:
        payload = msg.get_payload(decode=True)
        if payload:
            plain_body = payload.decode("utf-8", errors="ignore")

    if html_body.strip():
        extractor = HTMLTextExtractor()
        extractor.feed(html_body)
        body = extractor.get_text()
    elif plain_body.strip():
        body = plain_body
    else:
        body = ""

    body = re.sub(r"\s+", " ", body).strip()
    return subject, body


def prepare_email_text(subject, body):
    """Combine subject and body for model prediction.

    Args:
        subject: Email subject
        body: Email body

    Returns:
        Combined text string ready for prediction
    """
    eml_text = (str(subject) + " " + str(body)).lower()
    eml_text = re.sub(r"\s+", " ", eml_text).strip()
    return eml_text


def clean_text(text):
    """Full preprocessing pipeline for a single text string.

    Args:
        text: The raw text string.

    Returns:
        The cleaned text string.
    """
    if not isinstance(text, str):
        text = str(text)
    tp = TextPreprocessor(text)
    return (
        tp.strip_html_tags()
        .replace_urls()
        .replace_emails()
        .replace_phone_numbers()
        .replace_numbers()
        .replace_percentages()
        .replace_emojis()
        .remove_special_characters()
        .normalize_whitespace()
        .to_lowercase()
        .get_text()
    )
