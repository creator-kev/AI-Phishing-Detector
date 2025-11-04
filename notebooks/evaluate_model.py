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
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# Setup paths
# -----------------------------
SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[1]

DATA_PATH = PROJECT_ROOT / "data" / "phishing.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "phishing_pipeline.pkl"

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip().str.lower()

if "class" not in df.columns:
    raise KeyError(f"'class' column not found. Available: {df.columns.tolist()}")

if "index" in df.columns:
    df = df.drop(columns=["index"])

X = df.drop(columns=["class"])
y = df["class"]

# -----------------------------
# Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -----------------------------
# Load Model
# -----------------------------
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Trained model not found at {MODEL_PATH}")

model = joblib.load(MODEL_PATH)
print(f"✅ Loaded model from: {MODEL_PATH}")

# -----------------------------
# Evaluate
# -----------------------------
print(f"\nEvaluating model on {len(X_test)} test samples...")
y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

print("\n" + "=" * 60)
print("MODEL EVALUATION RESULTS")
print("=" * 60)
print(f"✅ Accuracy:  {acc:.4f} ({acc*100:.2f}%)")
print(f"✅ Precision: {prec:.4f} ({prec*100:.2f}%)")
print(f"✅ Recall:    {rec:.4f} ({rec*100:.2f}%)")
print(f"✅ F1-Score:  {f1:.4f}")
print("=" * 60 + "\n")

print("CLASSIFICATION REPORT:")
print("-" * 60)
print(classification_report(y_test, y_pred, zero_division=0))

# -----------------------------
# Confusion Matrix
# -----------------------------
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=["Legitimate", "Phishing"],
    yticklabels=["Legitimate", "Phishing"]
)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix - Phishing Detector")
plt.tight_layout()

output_path = PROJECT_ROOT / "confusion_matrix.png"
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"✅ Confusion matrix saved to: {output_path}")
plt.show()
