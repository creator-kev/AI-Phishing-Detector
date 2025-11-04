# notebooks/train_pipeline.py
import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# -----------------------------
# Paths
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "../data/phishing.csv")
MODEL_DIR = os.path.join(BASE_DIR, "../models")
MODEL_PATH = os.path.join(MODEL_DIR, "phishing_pipeline.pkl")

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip().str.lower()

if "class" not in df.columns:
    raise KeyError(f"'class' column not found. Available: {df.columns.tolist()}")

# Drop non-feature columns
if "index" in df.columns:
    df = df.drop(columns=["index"])

X = df.drop(columns=["class"])
y = df["class"]

# -----------------------------
# Split Data
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -----------------------------
# Build Pipeline
# -----------------------------
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))
])

# -----------------------------
# Train
# -----------------------------
print("Training numeric model...")
pipeline.fit(X_train, y_train)

# -----------------------------
# Evaluate
# -----------------------------
y_pred = pipeline.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"\n✅ Accuracy: {acc:.4f} ({acc*100:.2f}%)")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# -----------------------------
# Save Model
# -----------------------------
os.makedirs(MODEL_DIR, exist_ok=True)
joblib.dump(pipeline, MODEL_PATH)
print(f"\n✅ Saved trained model to: {MODEL_PATH}")
