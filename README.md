# Email Spam Detection

A machine learning project for binary email spam classification, comparing supervised (Logistic Regression, Complement Naive Bayes, Linear SVM) and unsupervised (K-Means) methods across 18 combined public datasets.

**Label convention:** `0` = ham (legitimate), `1` = spam

---

## Project Structure

```
├── src/                       # Reusable Python modules
│   ├── config.py              # Paths and directory setup
│   ├── load_data.py           # 15 dataset loader functions (raw CSV → [text, label])
│   ├── preprocess.py          # TextPreprocessor, HTML stripping, .eml parsing
│   ├── features.py            # extract_features() — 9 numeric email features
│   ├── kmeans_classifier.py   # Custom KMeansClassifier (sklearn-compatible wrapper)
│   └── test_emails.py         # 11 hardcoded test emails (6 ham, 5 spam)
├── notebooks/                 # Jupyter notebooks
│   ├── clean_*.ipynb          # Dataset-specific cleaning notebooks
│   ├── clean_combine.ipynb    # Merge cleaned datasets → cleaned_combined.csv
│   ├── clean_combine2.ipynb   # Extended merge → cleaned_combined_v2.csv
│   ├── processed_data_analysis.ipynb  # EDA and visualizations
│   ├── logistic_regression.ipynb      # Logistic Regression training
│   ├── naive_bayes.ipynb             # Complement Naive Bayes training
│   ├── linear_svc.ipynb              # Linear SVM training
│   ├── kmeans.ipynb                  # K-Means (basic)
│   ├── kmeans_combined.ipynb         # K-Means (custom KMeansClassifier)
│   ├── kmeans_combined_pipeline.ipynb# K-Means (sklearn Pipeline)
│   └── model_comparison.ipynb        # Unified evaluation of all models
├── models/                    # Saved trained models (*.pkl, gitignored)
│   ├── logistic_regression_pipeline.pkl
│   ├── naive_bayes_pipeline.pkl
│   ├── linear_svc_pipeline.pkl
│   ├── kmeans_combined.pkl
│   ├── kmeans_spam_model.pkl
│   └── kmeans_combined_pipeline.pkl
├── data/
│   ├── raw/                   # Original CSV datasets (gitignored)
│   └── processed/             # Cleaned/combined datasets (gitignored)
├── tests/
│   └── test_preprocess.py     # Pytest unit tests for TextPreprocessor
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

18 raw CSV datasets are expected in `data/raw/`. They are gitignored and must be obtained separately (see `src/config.py:DATA_PATHS` for file list).

### Further Data Processing

To process individual raw datasets into cleaned CSV files, run the corresponding notebooks in order:

```bash
# Step 1 — Clean individual datasets
jupyter notebook notebooks/clean_trec05.ipynb      # → cleaned_email_data_v1.csv
jupyter notebook notebooks/clean_ceas.ipynb        # → cleaned_ceas_08.csv
jupyter notebook notebooks/clean_spamassassin.ipynb # → cleaned_spamassassin.csv
jupyter notebook notebooks/clean_nazario.ipynb     # → cleaned_nazario.csv
jupyter notebook notebooks/clean_nigerian_fraud.ipynb # → cleaned_nigerian_fraud.csv
jupyter notebook notebooks/clean_trec07.ipynb      # → cleaned_trec07.csv

# Step 2 — Combine cleaned datasets
jupyter notebook notebooks/clean_combine.ipynb     # → cleaned_combined.csv (5 datasets)
jupyter notebook notebooks/clean_combine2.ipynb    # → cleaned_combined_v2.csv (6 datasets)
```

The final dataset used by all model notebooks is `data/processed/cleaned_combined_v3.csv` (created manually from v2).

Each cleaning notebook:

1. Loads a raw CSV via its loader function in `src/load_data.py`
2. Normalises columns to `[text, label]` (label: 0 = ham, 1 = spam)
3. Concatenates subject + body into the `text` field
4. Applies `extract_features()` from `src/features.py` (adds 9 numeric columns)
5. Applies `clean_db_text()` and `process_text_advanced()` from `src/preprocess.py`
6. Saves to `data/processed/cleaned_*.csv`

---

## Model Training

Each model notebook reads `data/processed/cleaned_combined_v3.csv`, splits 80/20 (stratified, `random_state=42`), builds a `Pipeline` with `TfidfVectorizer` + scaler + classifier, and saves the trained pipeline to `models/`.

```bash
# Train all models (run each notebook)
jupyter notebook notebooks/logistic_regression.ipynb  # → models/logistic_regression_pipeline.pkl
jupyter notebook notebooks/naive_bayes.ipynb           # → models/naive_bayes_pipeline.pkl
jupyter notebook notebooks/linear_svc.ipynb            # → models/linear_svc_pipeline.pkl
jupyter notebook notebooks/kmeans_combined.ipynb       # → models/kmeans_combined.pkl

# Compare all models
jupyter notebook notebooks/model_comparison.ipynb
```

Alternatively, training can be triggered programmatically:

```python
import joblib
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

df = pd.read_csv("data/processed/cleaned_combined_v3.csv")
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

| Model                  | Type           | Accuracy | Precision | Recall | F1     |
| ---------------------- | -------------- | -------- | --------- | ------ | ------ |
| Linear SVM             | Classification | 99.59%   | 99.66%    | 99.52% | 99.59% |
| Logistic Regression    | Classification | 99.21%   | 99.34%    | 99.09% | 99.22% |
| Complement Naive Bayes | Classification | 97.63%   | 98.57%    | 96.71% | 97.63% |
| K-Means (custom)       | Clustering     | 70.07%   | 82.45%    | 51.67% | 63.52% |

Trained on 121,084 samples (49.6% ham, 50.4% spam) from 9 combined datasets. Test set: 24,217 held-out samples.

---

## References

- Enron Email Dataset — https://www.cs.cmu.edu/~./enron/
- TREC Spam Corpus — https://plg.uwaterloo.ca/~gvcormac/treccorpus/
- SpamAssassin Corpus — https://spamassassin.apache.org/old/publiccorpus/
- CEAS 2008 Corpus — https://www.ceas.cc/
- Nazario Phishing Corpus — https://monkey.org/~jose/phishing/
- LingSpam Dataset — https://www.cs.uic.edu/~liub/FBS/sentiment-analysis.html
