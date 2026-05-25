# Email Spam Detection

A machine learning project for binary email spam classification, comparing supervised (Logistic Regression, Complement Naive Bayes, Linear SVM) and unsupervised (K-Means) methods across 18 public datasets.

**Label convention:** `0` = ham (legitimate), `1` = spam

---

## Project Structure

```
├── src/                       # Reusable Python modules
│   ├── config.py              # Paths and directory setup
│   ├── load_data.py           # 15 dataset loader functions (raw CSV → [text, label])
│   ├── preprocess.py          # TextPreprocessor, HTML stripping, .eml parsing
│   ├── features.py            # extract_features() — 9 numeric email features
│   └── test_emails.py         # 11 hardcoded test emails (6 ham, 5 spam)
├── notebooks/                 # Jupyter notebooks
│   ├── clean_*.ipynb          # Dataset-specific cleaning notebooks
│   ├── clean_combine.ipynb    # Merge cleaned datasets → cleaned_combined.csv
│   ├── processed_data_analysis.ipynb  # EDA and visualizations
│   ├── logistic_regression.ipynb      # Logistic Regression training
│   ├── naive_bayes.ipynb             # Complement Naive Bayes training
│   ├── linear_svm.ipynb              # Linear SVM training
│   ├── kmeans_combined_pipeline.ipynb# K-Means (sklearn Pipeline)
│   └── model_comparison.ipynb        # Unified evaluation of all models
├── models/                    # Saved trained models (*.pkl, gitignored)
│   ├── logistic_regression_pipeline.pkl
│   ├── naive_bayes_pipeline.pkl
│   ├── linear_svm_pipeline.pkl
│   └── kmeans_pipeline.pkl
├── data/
│   ├── raw/                   # 18 original CSV datasets (gitignored)
│   └── processed/             # Cleaned/combined datasets (gitignored)
├── tests/
│   └── test_preprocess.py     # Pytest unit tests for TextPreprocessor
├── scripts/
│   └── add_notebook_comments.py  # Utility to insert markdown cells
├── results/                   # Output directory (empty)
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

---

## Environment Setup

### Using conda (recommended)

```bash
conda create -n email-spam-detection python=3.11 -y
conda activate email-spam-detection
pip install -r requirements.txt
```

### Using venv (if conda unavailable)

```bash
python -m venv .venv
# Windows:
.venv\Scripts\Activate.ps1
# macOS / Linux:
# source .venv/bin/activate
pip install -r requirements.txt
```

The virtual environment directory (`.venv/`) is gitignored.

---

## Data Pipeline

### Raw Datasets

18 raw CSV datasets are expected in `data/raw/`. They are gitignored and must be obtained separately. The datasets include: Enron, Enron 2, SpamAssassin, SpamAssassin 2, email_spam_dataset, phishing_email, SMS spam, spam_email_detection, 250K+ email, 190K+ email, NLP spam/ham, Ling-Spam, Nigerian fraud, Nazario, CEAS 08, email_origin, email_text, and TREC 07.

### Processing Steps

```
raw/*.csv → clean_*_dataset.ipynb → cleaned_*.csv
                                       ↓
                              clean_combine.ipynb → cleaned_combined.csv
```

To process individual raw datasets into cleaned CSV files, run the corresponding notebooks:

```bash
# Step 1 — Clean individual datasets
jupyter lab notebooks/clean_trec05.ipynb      # → cleaned_email_data_v1.csv
jupyter lab notebooks/clean_ceas.ipynb        # → cleaned_ceas_08.csv
jupyter lab notebooks/clean_spamassassin.ipynb # → cleaned_spamassassin.csv
jupyter lab notebooks/clean_nazario.ipynb     # → cleaned_nazario.csv
jupyter lab notebooks/clean_nigerian_fraud.ipynb # → cleaned_nigerian_fraud.csv

# Step 2 — Combine cleaned datasets
jupyter lab notebooks/clean_combine.ipynb     # → cleaned_combined.csv
```

The final dataset used by all model notebooks is `data/processed/cleaned_combined.csv`.

Each cleaning notebook:

1. Loads a raw CSV via its loader function in `src/load_data.py`
2. Normalises columns to `[text, label]` (label: 0 = ham, 1 = spam)
3. Concatenates subject + body into the `text` field
4. Applies `extract_features()` from `src/features.py` (adds 9 numeric columns)
5. Applies `clean_db_text()` and `process_text_advanced()` from `src/preprocess.py`
6. Saves to `data/processed/cleaned_*.csv`

---

## Feature Engineering

The `extract_features()` function in `src/features.py` takes raw email text and produces a DataFrame with 9 numeric columns plus cleaned text:

| Feature           | Description                               |
| ----------------- | ----------------------------------------- |
| `num_urls`        | Count of URLs (http/https/www)            |
| `num_exclamation` | Count of `!` characters                   |
| `num_question`    | Count of `?` characters                   |
| `num_dollar`      | Count of `$` signs                        |
| `num_all_caps`    | Count of ALL-CAPS words (3+ chars)        |
| `num_numbers`     | Count of numeric tokens                   |
| `word_count`      | Total word count                          |
| `capital_ratio`   | Fraction of characters that are uppercase |
| `emoji_count`     | Count of emoji characters                 |

Text cleaning is performed by `clean_db_text()` (heavy cleaning with bracket/hex removal) followed by `process_text_advanced()` (stemming, stopword removal).

---

## Model Training

Each model notebook reads `data/processed/cleaned_combined.csv`, splits 80/20 (stratified, `random_state=42`), builds a `Pipeline` with `TfidfVectorizer` + classifier, and saves the trained pipeline to `models/`.

```bash
# Train all models (run each notebook)
jupyter lab notebooks/logistic_regression.ipynb  # → models/logistic_regression_pipeline.pkl
jupyter lab notebooks/naive_bayes.ipynb           # → models/naive_bayes_pipeline.pkl
jupyter lab notebooks/linear_svm.ipynb            # → models/linear_svm_pipeline.pkl
jupyter lab notebooks/kmeans_combined_pipeline.ipynb  # → models/kmeans_pipeline.pkl

# Compare all models
jupyter lab notebooks/model_comparison.ipynb
```

### Supervised Models

All three supervised models share the same pipeline pattern:

```python
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=50000, ngram_range=(1, 2), sublinear_tf=True)),
    ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))
])
pipeline.fit(X_train, y_train)
```

The classifiers differ:

- **Logistic Regression** — `LogisticRegression(max_iter=1000, class_weight="balanced")`
- **Complement Naive Bayes** — `ComplementNB()`
- **Linear SVM** — `LinearSVC(max_iter=2000, dual="auto")`

### K-Means (Unsupervised)

K-Means uses a Pipeline with TF-IDF vectorization and sklearn's `KMeans(n_clusters=2)`. After fitting, cluster labels (0/1) are mapped to class labels (0/1) via majority vote on training data. Probability scores are derived from the sigmoid of distance to the spam centroid:

```python
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        max_features=50000, sublinear_tf=True, ngram_range=(1, 2),
        max_df=0.7, min_df=5, stop_words="english",
    )),
    ("kmeans", KMeans(n_clusters=2, random_state=42, n_init=10)),
])
pipeline.fit(X_train)

# Post-fit cluster-to-label mapping
train_clusters = pipeline.predict(X_train)
map_0 = y_train[train_clusters == 0].mode()[0]
map_1 = y_train[train_clusters == 1].mode()[0]
```

### Programmatic Training

```python
import joblib
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

df = pd.read_csv("data/processed/cleaned_combined.csv")
X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
)

pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=50000, ngram_range=(1, 2), sublinear_tf=True)),
    ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))
])
pipeline.fit(X_train, y_train)

joblib.dump(pipeline, "models/lr_model.pkl")
print(f"Test accuracy: {pipeline.score(X_test, y_test):.4f}")
```

---

## Model Comparison

The `model_comparison.ipynb` notebook loads all four trained pipelines and evaluates them on an identical test split. For each model it reports accuracy, precision, recall, F1, and ROC-AUC, and generates bar charts, confusion matrices, ROC curves, and test-email predictions.

---

## Prediction

### Using a saved model on new text

```python
import joblib

pipeline = joblib.load("models/logistic_regression_pipeline.pkl")

emails = [
    "Congratulations! You won $1000 free cash prize. Click here now!",
    "Hi, can we reschedule tomorrow's meeting to 2pm?",
]

preds = pipeline.predict(emails)       # [1, 0]  (1 = spam, 0 = ham)
probs = pipeline.predict_proba(emails) # probabilities
```

### Using the built-in test harness

```python
import sys
from pathlib import Path
sys.path.append(str(Path.cwd()))
from src.test_emails import test_emails

import joblib
pipeline = joblib.load("models/logistic_regression_pipeline.pkl")

for i, email in enumerate(test_emails):
    pred = pipeline.predict([email])[0]
    label = "SPAM" if pred == 1 else "HAM"
    print(f"[{i}] {label}: {email[:60]}...")
```

---

## Testing

```bash
pytest tests/test_preprocess.py
```

---

## Results Summary

| Model                  | Type           | Accuracy | Precision | Recall | F1     | ROC-AUC |
| ---------------------- | -------------- | -------- | --------- | ------ | ------ | ------- |
| Linear SVM             | Classification | 99.55%   | 99.67%    | 99.44% | 99.55% | 99.99%  |
| Logistic Regression    | Classification | 99.14%   | 99.41%    | 98.90% | 99.15% | 99.96%  |
| Complement Naive Bayes | Classification | 97.79%   | 98.75%    | 96.86% | 97.80% | 99.77%  |
| K-Means                | Clustering     | 70.17%   | 83.26%    | 51.59% | 63.70% | 63.87%  |

Trained on 118,988 samples (50.7% spam) from cleaned_combined.csv. Test set: 23,798 held-out samples (20% stratified split).

**Key takeaways:**

- All supervised linear classifiers achieve >97% accuracy; Linear SVM is the best at 99.59%
- K-Means clustering (70.07%) cannot match supervised methods — it optimises distance to centroid, not classification accuracy
- The real challenge is generalising beyond 2000-era Enron/TREC data to modern email patterns

---

