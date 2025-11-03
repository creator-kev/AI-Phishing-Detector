# notebooks/train_pipeline.py
import re
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.compose import ColumnTransformer
from train_pipeline import URLFeatureExtractor
import joblib
import os



# ---- Config ----
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "../data/phishing.csv")  # ✅ correct spelling + path
MODEL_DIR = os.path.join(BASE_DIR, "../models")             # ✅ consistent naming
MODEL_PATH = os.path.join(MODEL_DIR, "phishing_pipeline.pkl")  # ✅ no space, consistent case




# ---- Utility feature extractor for URL/text ----
class URLFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Transformer that accepts a pandas Series or array of URL/text and returns
    a DataFrame/2D-array of numeric features derived from the text.
    """
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def _extract(self, text):
        if not isinstance(text, str):
            text = str(text)
        features = {}
        features['url_length'] = len(text)
        features['num_dots'] = text.count('.')
        features['num_slash'] = text.count('/')
        features['has_https'] = int('https' in text.lower())
        features['has_at'] = int('@' in text)
        features['has_ip'] = int(bool(re.search(r'\d+\.\d+\.\d+\.\d+', text)))
        features['has_login'] = int(bool(re.search(r'login|signin|sign-in', text, re.I)))
        features['has_verify'] = int(bool(re.search(r'verify|confirm|account', text, re.I)))
        features['has_bank'] = int(bool(re.search(r'bank|paypal|amazon|microsoft|apple', text, re.I)))
        return list(features.values())

    def transform(self, X):
        # X expected to be iterable of strings
        arr = [self._extract(x) for x in X]
        return np.array(arr)

# ---- Load dataset ----
df = pd.read_csv(DATA_PATH)
# Adjust these column names if different in your CSV
# Try to detect a URL/text column and a label column
text_col = "url" if "url" in df.columns else df.columns[0]
label_col = "Result" if "Result" in df.columns else df.columns[-1]

# Clean/convert label if needed (ensure binary 0/1)
# Example dataset might use 1 (phishing) and -1 (legit); convert accordingly
y = df[label_col].copy()
y = y.replace(-1, 0)  # if dataset uses -1 for legit

X_text = df[text_col].astype(str)
# ---- Train/test split ----
X_train, X_test, y_train, y_test = train_test_split(X_text, y, test_size=0.2, random_state=42, stratify=y)

# ---- Build pipeline ----
# TF-IDF pipeline for raw text
tfidf = Pipeline([
    ("tfidf_vect", TfidfVectorizer(ngram_range=(1,2), analyzer='char_wb', max_features=5000)),
])

# numeric features pipeline (from URLFeatureExtractor)
numeric_pipeline = Pipeline([
    ("url_feats", URLFeatureExtractor()),
    ("scaler", StandardScaler()),
])

# Combine features: TF-IDF (sparse) + numeric features (dense)
from sklearn.preprocessing import FunctionTransformer
tfidf_transformer = ("tfidf", tfidf)
numeric_transformer = ("num", numeric_pipeline)

# FeatureUnion expects each transformer to consume the same input (the raw text),
# so we pass the raw text to both transformers.
full = FeatureUnion([tfidf_transformer, numeric_transformer])

final_pipeline = Pipeline([
    ("features", full),
    ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))
])

# ---- Fit ----
print("Training pipeline...")
final_pipeline.fit(X_train, y_train)

# ---- Evaluate ----
y_pred = final_pipeline.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification report:\n", classification_report(y_test, y_pred))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

# ---- Save pipeline ----
os.makedirs(MODEL_DIR, exist_ok=True)
joblib.dump(final_pipeline, MODEL_PATH)
print(f"Saved pipeline to: {MODEL_PATH}")
