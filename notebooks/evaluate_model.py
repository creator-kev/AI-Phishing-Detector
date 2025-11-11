# notebooks/evaluate_model.py
import os
from pathlib import Path
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# ----- Setup paths -----
SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "phishing.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "phishing_pipeline.pkl"

print("="*60)
print("AI Phishing Detector - Model Evaluation")
print("="*60)

# ----- Load dataset -----
df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip()  # keep original casing

# ----- Find label column -----
label_col = None
for col in ["class", "label", "result", "target"]:
    if col in df.columns:
        label_col = col
        break
if label_col is None:
    raise KeyError("No label column found in the dataset. Expected one of: class, label, result, target")

# ----- Select feature columns -----
drop_cols = [label_col]
# Drop index/unnamed columns (case-insensitive)
for c in df.columns:
    if c.lower() in ["index", "unnamed: 0"]:
        drop_cols.append(c)

feature_cols = [c for c in df.columns if c not in drop_cols]
if len(feature_cols) == 0:
    raise KeyError("No feature columns found. Check your dataset")

X = df[feature_cols]
y = df[label_col].astype(int)

print(f"Features used ({len(feature_cols)}): {feature_cols[:10]}{'...' if len(feature_cols)>10 else ''}")
print(f"Total samples: {len(df)}")

# ----- Split dataset -----
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ----- Load model -----
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"❌ Model not found at {MODEL_PATH}")

model = joblib.load(MODEL_PATH)
print(f"✅ Loaded model from: {MODEL_PATH}")

# Optional: print model feature names for debugging
if hasattr(model, 'feature_names_in_'):
    print("Feature names expected by model:", model.feature_names_in_)

# ----- Evaluate -----
print(f"\nEvaluating on {len(X_test)} test samples...")
y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

print("\n" + "="*60)
print("MODEL EVALUATION RESULTS")
print("="*60)
print(f"✅ Accuracy:  {acc:.4f} ({acc*100:.2f}%)")
print(f"✅ Precision: {prec:.4f} ({prec*100:.2f}%)")
print(f"✅ Recall:    {rec:.4f} ({rec*100:.2f}%)")
print(f"✅ F1-Score:  {f1:.4f}")
print("="*60 + "\n")

print("CLASSIFICATION REPORT:")
print("-"*60)
print(classification_report(y_test, y_pred, target_names=['Legitimate', 'Phishing'], zero_division=0))

# ----- Confusion matrix -----
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=["Legitimate", "Phishing"],
    yticklabels=["Legitimate", "Phishing"]
)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix - AI Phishing Detector")
plt.tight_layout()

output_path = PROJECT_ROOT / "docs" / "confusion_matrix.png"
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"✅ Confusion matrix saved to: {output_path}")
print("\nEvaluation complete!")
