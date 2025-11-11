# notebooks/train_pipeline.py
"""
Train pipeline for AI Phishing Detector.

This script handles two dataset formats:
1) CSV contains a 'url' column -> we extract features from raw URLs (URLFeatureExtractor).
2) CSV already contains numeric/extracted features -> we train directly on those features.

The trained model (RandomForest) is saved to models/phishing_pipeline.pkl.
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import re
import warnings

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

warnings.filterwarnings("ignore")

# -----------------------
# Improved URLFeatureExtractor (used only if CSV has 'url' column)
# -----------------------
class URLFeatureExtractor(BaseEstimator, TransformerMixin):
    """Extract enhanced features from URLs"""
    def __init__(self):
        self.keywords = ['login','signin','bank','paypal','amazon','secure','account','update','verify']

    def fit(self, X, y=None):
        return self

    def _extract(self, url):
        if not isinstance(url, str):
            url = str(url)
        features = {}
        features['url_length'] = len(url)
        features['num_dots'] = url.count('.')
        features['num_slash'] = url.count('/')
        features['num_hyphen'] = url.count('-')
        features['num_digits'] = sum(c.isdigit() for c in url)
        features['has_https'] = int('https' in url.lower())
        features['has_at'] = int('@' in url)
        features['has_ip'] = int(bool(re.search(r'\d+\.\d+\.\d+\.\d+', url)))
        features['has_login'] = int(bool(re.search(r'login|signin|sign-in', url, re.I)))
        features['has_bank'] = int(bool(re.search(r'bank|paypal|amazon|microsoft|apple', url, re.I)))
        features['suspicious_keywords'] = int(any(k in url.lower() for k in self.keywords))

        # Subdomain count
        domain_part = url.split('//')[-1].split('/')[0]
        features['num_subdomains'] = len(domain_part.split('.')) - 2 if len(domain_part.split('.')) > 2 else 0

        # TLD length
        try:
            tld = domain_part.split('.')[-1]
            features['tld_length'] = len(tld)
        except Exception:
            features['tld_length'] = 0

        return list(features.values())

    def transform(self, X):
        if hasattr(X, 'values'):
            X = X.values
        return np.array([self._extract(x) for x in X])

# -----------------------
# Paths & config
# -----------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "phishing.csv"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODEL_DIR / "phishing_pipeline.pkl"
RSEED = 42

print("="*60)
print("AI Phishing Detector - Model Training")
print("="*60)
print(f"Project Root: {ROOT}")
print(f"Data path: {DATA_PATH}")
print(f"Model will be saved to: {MODEL_PATH}")

# -----------------------
# Load dataset
# -----------------------
if not DATA_PATH.exists():
    raise FileNotFoundError(f"Dataset not found at: {DATA_PATH}")

df = pd.read_csv(DATA_PATH)
# strip whitespace from column names but preserve case as-is
df.columns = df.columns.str.strip()

print(f"\n✅ Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"Columns (first 12): {df.columns.tolist()[:12]}")

# -----------------------
# Find label column
# -----------------------
label_col = None
for col in ["class", "Class", "label", "Label", "result", "target"]:
    if col in df.columns:
        label_col = col
        break
if label_col is None:
    raise KeyError(f"Label column not found. Available columns: {df.columns.tolist()}")

print(f"✅ Using label column: '{label_col}'")

# -----------------------
# Decide training mode: raw-URL or precomputed features
# -----------------------
use_url_column = any(c.lower() == "url" for c in df.columns)
if use_url_column:
    url_col = next(c for c in df.columns if c.lower() == "url")
    print(f"Detected raw URL column: '{url_col}' -> training from raw URLs with extractor.")
else:
    url_col = None
    print("No raw URL column detected -> training on existing numeric/extracted features.")

# -----------------------
# Prepare X, y
# -----------------------
y = df[label_col].astype(int)

if url_col:
    # Train using URLFeatureExtractor + RandomForest inside a Pipeline
    X = df[url_col].astype(str)
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RSEED, stratify=y
    )

    pipeline = Pipeline([
        ("features", URLFeatureExtractor()),
        ("clf", RandomForestClassifier(
            n_estimators=500,
            max_depth=None,
            random_state=RSEED,
            n_jobs=-1,
            class_weight='balanced'
        ))
    ])

    print(f"\n✅ Training using URLFeatureExtractor on raw URLs. Train samples: {len(X_train)}, Test samples: {len(X_test)}")

    pipeline.fit(X_train, y_train)

    # Evaluate
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n🎯 Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Legitimate', 'Phishing'], zero_division=0))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Save pipeline (it contains extractor + clf)
    joblib.dump(pipeline, MODEL_PATH)
    print(f"\n✅ Pipeline saved to: {MODEL_PATH}")

else:
    # Train directly on existing features
    # Drop label and index-like columns
    drop_cols = [label_col]
    for c in df.columns:
        if c.lower() in ["index", "unnamed: 0"]:
            drop_cols.append(c)

    feature_cols = [c for c in df.columns if c not in drop_cols]
    if len(feature_cols) == 0:
        raise KeyError("No feature columns found after dropping label/index columns. Check your CSV.")

    X = df[feature_cols].copy()

    # Ensure numeric: convert columns to numeric where possible, fill NaN with 0
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors='coerce').fillna(0)

    print(f"\n✅ Training on existing features ({len(feature_cols)}): {feature_cols[:12]}{'...' if len(feature_cols)>12 else ''}")

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RSEED, stratify=y
    )
    print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")

    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        random_state=RSEED,
        n_jobs=-1,
        class_weight='balanced'
    )

    clf.fit(X_train, y_train)

    # Evaluate
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n🎯 Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Legitimate', 'Phishing'], zero_division=0))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Feature importances
    try:
        importances = clf.feature_importances_
        fi = sorted(zip(feature_cols, importances), key=lambda x: x[1], reverse=True)
        print("\nTop features by importance:")
        for name, imp in fi[:12]:
            print(f"  {name}: {imp:.4f}")
    except Exception:
        pass

    # Save classifier (trained on DataFrame features) - classifier will have feature_names_in_
    joblib.dump(clf, MODEL_PATH)
    print(f"\n✅ Model saved to: {MODEL_PATH}")

# Done
print("\n🎉 Training complete. You can now run notebooks/evaluate_model.py or the Flask app.")
