# notebooks/evaluate_model.py
"""
AI Phishing Detector - Model Evaluation Script
==============================================
Comprehensive evaluation including:
- Accuracy, Precision, Recall, F1-Score
- Confusion Matrix visualization
- Feature importance analysis
- ROC curve and AUC
- Sample predictions
- Detailed classification report
"""


import sys
from pathlib import Path

# Ensure project root is in Python path
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))


import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
    roc_curve, auc, roc_auc_score
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from app.feature_extractor import StandardURLFeatureExtractor
# -----------------------
# Configuration
# -----------------------
SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "phishing.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "best_model.pkl"
DOCS_PATH = PROJECT_ROOT / "docs"

# Create docs directory
DOCS_PATH.mkdir(parents=True, exist_ok=True)

# Styling
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

print("\n" + "="*70)
print("🛡️  AI PHISHING DETECTOR - MODEL EVALUATION")
print("="*70)
print(f"📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"📁 Project Root: {PROJECT_ROOT}")
print(f"📊 Data: {DATA_PATH}")
print(f"🤖 Model: {MODEL_PATH}")
print("="*70 + "\n")

# -----------------------
# Load Dataset
# -----------------------
print("📂 Loading dataset...")
if not DATA_PATH.exists():
    print(f"❌ ERROR: Dataset not found at {DATA_PATH}")
    print("Please run: python generate_sample_dataset.py")
    sys.exit(1)

df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip()

print(f"✅ Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"📋 Columns: {df.columns.tolist()}\n")

# -----------------------
# Find Label Column
# -----------------------
label_col = None
for col in ["class", "Class", "label", "Label", "result", "Result", "target"]:
    if col in df.columns:
        label_col = col
        break

if label_col is None:
    print(f"❌ ERROR: No label column found")
    print(f"Available columns: {df.columns.tolist()}")
    sys.exit(1)

print(f"🏷️  Label column: '{label_col}'")

# -----------------------
# Prepare Data
# -----------------------
# Normalize labels to 0/1
y = df[label_col].astype(int)
unique_labels = sorted(y.unique())
print(f"📊 Label distribution: {unique_labels}")

if set(unique_labels) == {-1, 1}:
    y = y.replace({-1: 0, 1: 1})
    print("🔄 Converted labels: -1 → 0 (legitimate), 1 → 1 (phishing)")
elif set(unique_labels) != {0, 1}:
    print(f"⚠️  Warning: Unexpected label values: {unique_labels}")

# Determine feature columns
has_url = any(c.lower() == "url" for c in df.columns)

if has_url:
    url_col = next(c for c in df.columns if c.lower() == "url")
    X = df[url_col].astype(str)
    print(f"✅ Using raw URL column: '{url_col}'")
else:
    drop_cols = [label_col]
    for c in df.columns:
        if c.lower() in ["index", "unnamed: 0"]:
            drop_cols.append(c)
    
    feature_cols = [c for c in df.columns if c not in drop_cols]
    X = df[feature_cols].copy()
    
    # Convert to numeric
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors='coerce').fillna(0)
    
    print(f"✅ Using {len(feature_cols)} features: {feature_cols[:10]}")

print(f"📊 Class distribution:")
print(f"   Legitimate (0): {(y == 0).sum()} ({(y == 0).sum() / len(y) * 100:.1f}%)")
print(f"   Phishing (1):   {(y == 1).sum()} ({(y == 1).sum() / len(y) * 100:.1f}%)\n")

# -----------------------
# Split Data
# -----------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"📊 Data split:")
print(f"   Training:   {len(X_train)} samples")
print(f"   Testing:    {len(X_test)} samples\n")

# -----------------------
# Load Model
# -----------------------
print("🤖 Loading trained model...")
if not MODEL_PATH.exists():
    print(f"❌ ERROR: Model not found at {MODEL_PATH}")
    print("Please run: python notebooks/train_pipeline.py")
    sys.exit(1)

model = joblib.load(MODEL_PATH)
print(f"✅ Model loaded successfully\n")

# Optional: Display model info
if hasattr(model, 'feature_names_in_'):
    print(f"📋 Model features: {list(model.feature_names_in_)[:10]}...")
elif hasattr(model, 'named_steps'):
    print(f"📋 Model pipeline: {list(model.named_steps.keys())}")

# -----------------------
# Make Predictions
# -----------------------
print("\n" + "="*70)
print("🎯 MAKING PREDICTIONS")
print("="*70 + "\n")

y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probability of phishing (class 1)

# -----------------------
# Calculate Metrics
# -----------------------
print("📊 PERFORMANCE METRICS")
print("-"*70)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

try:
    auc_score = roc_auc_score(y_test, y_pred_proba)
except:
    auc_score = 0.0

print(f"✅ Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"✅ Precision: {precision:.4f} ({precision*100:.2f}%)")
print(f"✅ Recall:    {recall:.4f} ({recall*100:.2f}%)")
print(f"✅ F1-Score:  {f1:.4f}")
print(f"✅ ROC AUC:   {auc_score:.4f}")
print("-"*70 + "\n")

# Interpretation
print("📖 METRIC INTERPRETATIONS:")
print(f"   • Accuracy:  Overall correctness - {accuracy*100:.1f}% of predictions are correct")
print(f"   • Precision: Of predicted phishing, {precision*100:.1f}% are actually phishing")
print(f"   • Recall:    Of actual phishing, {recall*100:.1f}% are detected")
print(f"   • F1-Score:  Balance between precision and recall")
print(f"   • ROC AUC:   Model's ability to distinguish classes (1.0 = perfect)\n")

# -----------------------
# Classification Report
# -----------------------
print("="*70)
print("📋 DETAILED CLASSIFICATION REPORT")
print("="*70)
print(classification_report(
    y_test, y_pred, 
    target_names=['Legitimate', 'Phishing'],
    digits=4,
    zero_division=0
))

# -----------------------
# Confusion Matrix
# -----------------------
print("="*70)
print("📊 CONFUSION MATRIX")
print("="*70)

cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

print(f"\n                Predicted")
print(f"              Legit  Phish")
print(f"Actual Legit   {tn:4d}   {fp:4d}")
print(f"       Phish   {fn:4d}   {tp:4d}\n")

print(f"Breakdown:")
print(f"   • True Negatives (TN):  {tn} - Correctly identified as legitimate")
print(f"   • False Positives (FP): {fp} - Legitimate marked as phishing")
print(f"   • False Negatives (FN): {fn} - Phishing marked as legitimate")
print(f"   • True Positives (TP):  {tp} - Correctly identified as phishing\n")

if fp > 0:
    print(f"⚠️  {fp} legitimate URL(s) incorrectly flagged as phishing")
if fn > 0:
    print(f"⚠️  {fn} phishing URL(s) missed (marked as legitimate)")
print()

# -----------------------
# Visualizations
# -----------------------
print("="*70)
print("🎨 GENERATING VISUALIZATIONS")
print("="*70 + "\n")

# 1. Confusion Matrix Heatmap
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=["Legitimate", "Phishing"],
    yticklabels=["Legitimate", "Phishing"],
    cbar_kws={'label': 'Count'},
    annot_kws={'size': 16, 'weight': 'bold'}
)
plt.xlabel("Predicted Label", fontsize=12, weight='bold')
plt.ylabel("True Label", fontsize=12, weight='bold')
plt.title("Confusion Matrix - AI Phishing Detector", fontsize=14, weight='bold', pad=20)

# Add accuracy text
accuracy_text = f"Overall Accuracy: {accuracy:.2%}"
plt.text(1, -0.3, accuracy_text, ha='center', va='top', 
         fontsize=12, weight='bold', transform=ax.transAxes)

plt.tight_layout()
cm_path = DOCS_PATH / "confusion_matrix.png"
plt.savefig(cm_path, dpi=300, bbox_inches='tight')
print(f"✅ Confusion matrix saved: {cm_path}")
plt.close()

# 2. ROC Curve
try:
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {auc_score:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12, weight='bold')
    plt.ylabel('True Positive Rate', fontsize=12, weight='bold')
    plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=14, weight='bold', pad=20)
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    roc_path = DOCS_PATH / "roc_curve.png"
    plt.savefig(roc_path, dpi=300, bbox_inches='tight')
    print(f"✅ ROC curve saved: {roc_path}")
    plt.close()
except Exception as e:
    print(f"⚠️  Could not generate ROC curve: {e}")

# 3. Metrics Comparison Bar Chart
fig, ax = plt.subplots(figsize=(10, 6))
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC AUC']
values = [accuracy, precision, recall, f1, auc_score]
colors = ['#667eea', '#764ba2', '#f093fb', '#4facfe', '#43e97b']

bars = plt.bar(metrics, values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
plt.ylim([0, 1.1])
plt.ylabel('Score', fontsize=12, weight='bold')
plt.title('Model Performance Metrics', fontsize=14, weight='bold', pad=20)
plt.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bar, value in zip(bars, values):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
             f'{value:.3f}\n({value*100:.1f}%)',
             ha='center', va='bottom', fontsize=10, weight='bold')

plt.tight_layout()
metrics_path = DOCS_PATH / "metrics_comparison.png"
plt.savefig(metrics_path, dpi=300, bbox_inches='tight')
print(f"✅ Metrics comparison saved: {metrics_path}")
plt.close()

# 4. Feature Importance (if available)
try:
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        feature_names = model.feature_names_in_ if hasattr(model, 'feature_names_in_') else [f"Feature {i}" for i in range(len(importances))]
    elif hasattr(model, 'named_steps') and hasattr(model.named_steps.get('classifier', None), 'feature_importances_'):
        clf = model.named_steps['classifier']
        importances = clf.feature_importances_
        
        # Get feature names from extractor
        if hasattr(model.named_steps.get('feature_extractor', None), 'FEATURE_NAMES'):
            feature_names = model.named_steps['feature_extractor'].FEATURE_NAMES
        else:
            feature_names = [f"Feature {i}" for i in range(len(importances))]
    else:
        importances = None
    
    if importances is not None:
        # Create DataFrame and sort
        fi_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        # Plot top features
        top_n = min(15, len(fi_df))
        fig, ax = plt.subplots(figsize=(12, 8))
        
        colors_grad = plt.cm.viridis(np.linspace(0.3, 0.9, top_n))
        bars = plt.barh(range(top_n), fi_df['importance'].head(top_n), 
                       color=colors_grad, edgecolor='black', linewidth=1)
        plt.yticks(range(top_n), fi_df['feature'].head(top_n))
        plt.xlabel('Importance Score', fontsize=12, weight='bold')
        plt.title(f'Top {top_n} Feature Importances', fontsize=14, weight='bold', pad=20)
        plt.gca().invert_yaxis()
        plt.grid(axis='x', alpha=0.3)
        
        # Add value labels
        for i, (bar, value) in enumerate(zip(bars, fi_df['importance'].head(top_n))):
            plt.text(value + 0.002, i, f'{value:.4f}', 
                    va='center', fontsize=9, weight='bold')
        
        plt.tight_layout()
        fi_path = DOCS_PATH / "feature_importance.png"
        plt.savefig(fi_path, dpi=300, bbox_inches='tight')
        print(f"✅ Feature importance saved: {fi_path}")
        plt.close()
        
        # Print top features
        print(f"\n🔝 TOP 10 MOST IMPORTANT FEATURES:")
        print("-"*70)
        for idx, row in fi_df.head(10).iterrows():
            print(f"   {row['feature']:20s} : {row['importance']:.4f}")
        print()
        
except Exception as e:
    print(f"⚠️  Could not generate feature importance: {e}\n")

# 5. Prediction Distribution
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Histogram of predicted probabilities
ax1.hist(y_pred_proba[y_test == 0], bins=20, alpha=0.7, label='Legitimate', color='green', edgecolor='black')
ax1.hist(y_pred_proba[y_test == 1], bins=20, alpha=0.7, label='Phishing', color='red', edgecolor='black')
ax1.axvline(x=0.5, color='black', linestyle='--', linewidth=2, label='Threshold (0.5)')
ax1.set_xlabel('Predicted Probability of Phishing', fontsize=11, weight='bold')
ax1.set_ylabel('Frequency', fontsize=11, weight='bold')
ax1.set_title('Distribution of Predicted Probabilities', fontsize=12, weight='bold')
ax1.legend()
ax1.grid(alpha=0.3)

# Box plot of probabilities by true class
data_box = pd.DataFrame({
    'Probability': y_pred_proba,
    'True Class': ['Legitimate' if x == 0 else 'Phishing' for x in y_test]
})
sns.boxplot(x='True Class', y='Probability', data=data_box, ax=ax2, palette=['green', 'red'])
ax2.axhline(y=0.5, color='black', linestyle='--', linewidth=2, label='Threshold')
ax2.set_ylabel('Predicted Probability', fontsize=11, weight='bold')
ax2.set_title('Probability Distribution by True Class', fontsize=12, weight='bold')
ax2.legend()
ax2.grid(alpha=0.3)

plt.tight_layout()
dist_path = DOCS_PATH / "prediction_distribution.png"
plt.savefig(dist_path, dpi=300, bbox_inches='tight')
print(f"✅ Prediction distribution saved: {dist_path}")
plt.close()

# -----------------------
# Sample Predictions
# -----------------------
print("\n" + "="*70)
print("🔍 SAMPLE PREDICTIONS")
print("="*70 + "\n")

# Get some sample predictions
n_samples = min(10, len(X_test))
sample_indices = np.random.choice(len(X_test), n_samples, replace=False)

print(f"Showing {n_samples} random test samples:\n")

for i, idx in enumerate(sample_indices, 1):
    true_label = "Phishing" if y_test.iloc[idx] == 1 else "Legitimate"
    pred_label = "Phishing" if y_pred[idx] == 1 else "Legitimate"
    prob = y_pred_proba[idx]
    correct = "✅" if y_test.iloc[idx] == y_pred[idx] else "❌"
    
    # Get URL if available
    if has_url:
        url = X_test.iloc[idx][:60]
    else:
        url = "Feature-based prediction"
    
    print(f"{i:2d}. {correct} True: {true_label:12s} | Pred: {pred_label:12s} | Prob: {prob:.3f}")
    print(f"    URL: {url}")
    print()

# -----------------------
# Error Analysis
# -----------------------
print("="*70)
print("🔬 ERROR ANALYSIS")
print("="*70 + "\n")

# False Positives (Legitimate marked as Phishing)
fp_indices = np.where((y_test == 0) & (y_pred == 1))[0]
if len(fp_indices) > 0:
    print(f"⚠️  FALSE POSITIVES ({len(fp_indices)} cases):")
    print("   Legitimate URLs incorrectly marked as phishing:\n")
    for i, idx in enumerate(fp_indices[:5], 1):
        if has_url:
            url = X_test.iloc[idx]
            print(f"   {i}. {url[:70]}")
            print(f"      Probability: {y_pred_proba[idx]:.3f}\n")
else:
    print("✅ No false positives!\n")

# False Negatives (Phishing marked as Legitimate)
fn_indices = np.where((y_test == 1) & (y_pred == 0))[0]
if len(fn_indices) > 0:
    print(f"⚠️  FALSE NEGATIVES ({len(fn_indices)} cases):")
    print("   Phishing URLs incorrectly marked as legitimate:\n")
    for i, idx in enumerate(fn_indices[:5], 1):
        if has_url:
            url = X_test.iloc[idx]
            print(f"   {i}. {url[:70]}")
            print(f"      Probability: {y_pred_proba[idx]:.3f}\n")
else:
    print("✅ No false negatives!\n")

# -----------------------
# Save Evaluation Report
# -----------------------
print("="*70)
print("💾 SAVING EVALUATION REPORT")
print("="*70 + "\n")

report_path = DOCS_PATH / "evaluation_report.txt"
with open(report_path, 'w') as f:
    f.write("AI PHISHING DETECTOR - EVALUATION REPORT\n")
    f.write("="*70 + "\n")
    f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Model: {MODEL_PATH}\n")
    f.write(f"Dataset: {DATA_PATH}\n")
    f.write(f"Test Samples: {len(X_test)}\n\n")
    
    f.write("PERFORMANCE METRICS\n")
    f.write("-"*70 + "\n")
    f.write(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)\n")
    f.write(f"Precision: {precision:.4f} ({precision*100:.2f}%)\n")
    f.write(f"Recall:    {recall:.4f} ({recall*100:.2f}%)\n")
    f.write(f"F1-Score:  {f1:.4f}\n")
    f.write(f"ROC AUC:   {auc_score:.4f}\n\n")
    
    f.write("CONFUSION MATRIX\n")
    f.write("-"*70 + "\n")
    f.write(f"True Negatives:  {tn}\n")
    f.write(f"False Positives: {fp}\n")
    f.write(f"False Negatives: {fn}\n")
    f.write(f"True Positives:  {tp}\n\n")
    
    f.write("CLASSIFICATION REPORT\n")
    f.write("-"*70 + "\n")
    f.write(classification_report(y_test, y_pred, 
                                 target_names=['Legitimate', 'Phishing'],
                                 zero_division=0))

print(f"✅ Evaluation report saved: {report_path}")

# -----------------------
# Summary
# -----------------------
print("\n" + "="*70)
print("📊 EVALUATION SUMMARY")
print("="*70)
print(f"\n✅ Overall Accuracy: {accuracy:.2%}")
print(f"✅ Files generated: {len(list(DOCS_PATH.glob('*.png')))} visualizations")
print(f"✅ Report saved: {report_path}")
print(f"\n📁 All files saved in: {DOCS_PATH}")
print("\n" + "="*70)
print("🎉 EVALUATION COMPLETE!")
print("="*70 + "\n")
