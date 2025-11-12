# notebooks/train_pipeline.py
"""
FIXED: AI Phishing Detector Training Pipeline
- Handles both URL-based and feature-based datasets
- Ensures consistent feature extraction between training and inference
- Proper label handling for -1/1 and 0/1 encodings
"""

import sys
from pathlib import Path 


# Add project root to sys.path
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))


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
from app.feature_extractor import StandardURLFeatureExtractor
warnings.filterwarnings("ignore")




# -----------------------
# CRITICAL: Standardized Feature Extractor
# -----------------------
class StandardURLFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Extracts EXACTLY the same features as main.py's build_feature_vector()
    This ensures training/inference consistency.
    """
    
    FEATURE_NAMES = [
        'UsingIP', 'LongURL', 'ShortURL', 'Symbol@', 'Redirecting//',
        'PrefixSuffix-', 'SubDomains', 'HTTPS', 'DomainRegLen', 
        'InfoEmail', 'AbnormalURL'
    ]
    
    def fit(self, X, y=None):
        return self
    
    def _extract_features(self, url):
        """Extract features matching main.py exactly"""
        if not isinstance(url, str):
            url = str(url)
        
        txt = url.strip().lower()
        
        # Extract domain
        domain = ""
        if "://" in txt:
            domain = txt.split("://", 1)[1].split("/", 1)[0]
        else:
            domain = txt.split("/", 1)[0]
        
        features = {
            'UsingIP': int(bool(re.search(r'\d+\.\d+\.\d+\.\d+', txt))),
            'LongURL': len(txt),
            'ShortURL': int(len(txt) < 25),
            'Symbol@': int("@" in txt),
            'Redirecting//': int(txt.count("//") > 1),
            'PrefixSuffix-': int("-" in domain),
            'SubDomains': max(0, domain.count(".") - 1),
            'HTTPS': int(txt.startswith("https")),
            'DomainRegLen': len(domain),
            'InfoEmail': int(bool(re.search(r'[\w\.-]+@[\w\.-]+', txt))),
            'AbnormalURL': int(bool(re.search(
                r'(verify|secure|account|bank|update|login|signin|confirm|wp-admin)', 
                txt
            ))),
        }
        
        return features
    
    def transform(self, X):
        if hasattr(X, 'values'):
            X = X.values
        
        features_list = []
        for url in X:
            feat_dict = self._extract_features(url)
            # Ensure correct order
            features_list.append([feat_dict[name] for name in self.FEATURE_NAMES])
        
        return np.array(features_list)
    
    def get_feature_names_out(self, input_features=None):
        return np.array(self.FEATURE_NAMES)


# -----------------------
# Paths & Config
# -----------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "phishing.csv"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODEL_DIR / "phishing_pipeline.pkl"
RSEED = 42

print("="*60)
print("🛡️  AI Phishing Detector - FIXED Training Pipeline")
print("="*60)
print(f"📁 Data: {DATA_PATH}")
print(f"💾 Model: {MODEL_PATH}")

# -----------------------
# Load Dataset
# -----------------------
if not DATA_PATH.exists():
    raise FileNotFoundError(f"❌ Dataset not found: {DATA_PATH}")

df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip()

print(f"\n✅ Loaded: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"📋 Columns: {df.columns.tolist()}")

# -----------------------
# Find Label Column
# -----------------------
label_col = None
for col in ["Result", "Class", "class", "label", "Label", "result", "target"]:
    if col in df.columns:
        label_col = col
        break

if label_col is None:
    raise KeyError(f"❌ No label column found. Available: {df.columns.tolist()}")

print(f"🏷️  Label column: '{label_col}'")

# -----------------------
# Normalize Labels: Convert -1/1 to 0/1
# -----------------------
y_raw = df[label_col].astype(int)
unique_labels = sorted(y_raw.unique())
print(f"📊 Original labels: {unique_labels}")

if set(unique_labels) == {-1, 1}:
    # Convert -1 (legitimate) -> 0, 1 (phishing) -> 1
    y = y_raw.replace({-1: 0, 1: 1})
    print("🔄 Converted labels: -1 → 0 (legitimate), 1 → 1 (phishing)")
elif set(unique_labels) == {0, 1}:
    y = y_raw
    print("✅ Labels already in 0/1 format")
else:
    raise ValueError(f"❌ Unexpected label values: {unique_labels}. Expected {{-1,1}} or {{0,1}}")

# -----------------------
# Training Mode Detection
# -----------------------
has_url_col = any(c.lower() == "url" for c in df.columns)

if has_url_col:
    # MODE 1: Train from raw URLs
    url_col = next(c for c in df.columns if c.lower() == "url")
    X = df[url_col].astype(str)
    
    print(f"\n🌐 MODE: Training from raw URLs (column: '{url_col}')")
    print(f"📝 Sample URLs:\n{X.head(3).tolist()}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RSEED, stratify=y
    )
    
    # Build pipeline with standardized extractor
    pipeline = Pipeline([
        ("feature_extractor", StandardURLFeatureExtractor()),
        ("classifier", RandomForestClassifier(
            n_estimators=500,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=RSEED,
            n_jobs=-1,
            class_weight='balanced'
        ))
    ])
    
    print(f"🎯 Training samples: {len(X_train)} | Test samples: {len(X_test)}")
    print("⚙️  Training RandomForest with URL feature extraction...")
    
    pipeline.fit(X_train, y_train)
    
    # Evaluate
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    print(f"\n{'='*60}")
    print(f"📊 RESULTS")
    print(f"{'='*60}")
    print(f"🎯 Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    print(f"\n📈 Classification Report:")
    print(classification_report(y_test, y_pred, 
                                target_names=['Legitimate', 'Phishing'], 
                                zero_division=0))
    print(f"\n🔢 Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # Save pipeline
    joblib.dump(pipeline, MODEL_PATH)
    print(f"\n✅ Pipeline saved: {MODEL_PATH}")
    
else:
    # MODE 2: Train from pre-extracted features
    print(f"\n📊 MODE: Training from existing features")
    
    # Drop non-feature columns
    drop_cols = [label_col]
    for c in df.columns:
        if c.lower() in ["index", "unnamed: 0"]:
            drop_cols.append(c)
    
    feature_cols = [c for c in df.columns if c not in drop_cols]
    
    if len(feature_cols) == 0:
        raise KeyError("❌ No feature columns found")
    
    X = df[feature_cols].copy()
    
    # Convert to numeric
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors='coerce').fillna(0)
    
    print(f"📋 Features ({len(feature_cols)}): {feature_cols}")
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RSEED, stratify=y
    )
    
    print(f"🎯 Training samples: {len(X_train)} | Test samples: {len(X_test)}")
    
    # Train classifier
    clf = RandomForestClassifier(
        n_estimators=500,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=RSEED,
        n_jobs=-1,
        class_weight='balanced'
    )
    
    print("⚙️  Training RandomForest...")
    clf.fit(X_train, y_train)
    
    # Evaluate
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    print(f"\n{'='*60}")
    print(f"📊 RESULTS")
    print(f"{'='*60}")
    print(f"🎯 Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    print(f"\n📈 Classification Report:")
    print(classification_report(y_test, y_pred, 
                                target_names=['Legitimate', 'Phishing'], 
                                zero_division=0))
    print(f"\n🔢 Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # Feature importance
    importances = clf.feature_importances_
    fi = sorted(zip(feature_cols, importances), key=lambda x: x[1], reverse=True)
    print(f"\n🔝 Top 10 Features:")
    for name, imp in fi[:10]:
        print(f"  • {name}: {imp:.4f}")
    
    # Save model
    joblib.dump(clf, MODEL_PATH)
    print(f"\n✅ Model saved: {MODEL_PATH}")

print(f"\n{'='*60}")
print("🎉 Training Complete!")
print("📌 Next steps:")
print("   1. Run: python notebooks/evaluate_model.py")
print("   2. Start Flask app: python app/main.py")
print(f"{'='*60}\n")
