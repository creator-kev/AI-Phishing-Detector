# notebooks/compare_models.py
"""
Compare multiple ML models with cross-validation and hyperparameter tuning.
Tests: Random Forest, XGBoost, Gradient Boosting, Logistic Regression
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import time
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score
)

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠️  XGBoost not installed. Install with: pip install xgboost")

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# === CONFIGURATION ===
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "phishing_features.csv"
MODEL_DIR = ROOT / "models"
DOCS_DIR = ROOT / "docs"

MODEL_DIR.mkdir(exist_ok=True)
DOCS_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5


def load_data():
    """Load feature dataset"""
    print(f"Loading data from: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    
    if 'class' not in df.columns:
        raise ValueError("Dataset must contain 'class' column")
    
    X = df.drop('class', axis=1)
    y = df['class']
    
    print(f"✅ Loaded {len(df)} samples with {X.shape[1]} features")
    print(f"   Class distribution: {y.value_counts().to_dict()}")
    
    return X, y


def evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    """Train and evaluate a single model"""
    print(f"\n{'='*60}")
    print(f"Training: {model_name}")
    print(f"{'='*60}")
    
    # Train
    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    # Predict
    y_pred = model.predict(X_test)
    
    # Get probabilities if available
    try:
        y_proba = model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_proba)
    except:
        roc_auc = None
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=CV_FOLDS, scoring='accuracy')
    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()
    
    # Results
    results = {
        'model_name': model_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'cv_mean': cv_mean,
        'cv_std': cv_std,
        'train_time': train_time,
        'predictions': y_pred,
        'model_object': model
    }
    
    print(f"✅ Accuracy:   {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"✅ Precision:  {precision:.4f}")
    print(f"✅ Recall:     {recall:.4f}")
    print(f"✅ F1-Score:   {f1:.4f}")
    if roc_auc:
        print(f"✅ ROC-AUC:    {roc_auc:.4f}")
    print(f"✅ CV Score:   {cv_mean:.4f} (+/- {cv_std:.4f})")
    print(f"⏱️  Train Time: {train_time:.2f}s")
    
    return results


def plot_comparison(results_list, output_path):
    """Create comparison plots for all models"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    models = [r['model_name'] for r in results_list]
    
    # 1. Accuracy comparison
    accuracies = [r['accuracy'] for r in results_list]
    axes[0, 0].barh(models, accuracies, color='skyblue')
    axes[0, 0].set_xlabel('Accuracy')
    axes[0, 0].set_title('Model Accuracy Comparison')
    axes[0, 0].set_xlim([0, 1])
    
    # 2. F1-Score comparison
    f1_scores = [r['f1_score'] for r in results_list]
    axes[0, 1].barh(models, f1_scores, color='lightgreen')
    axes[0, 1].set_xlabel('F1-Score')
    axes[0, 1].set_title('Model F1-Score Comparison')
    axes[0, 1].set_xlim([0, 1])
    
    # 3. Precision vs Recall
    precisions = [r['precision'] for r in results_list]
    recalls = [r['recall'] for r in results_list]
    
    x = np.arange(len(models))
    width = 0.35
    axes[1, 0].bar(x - width/2, precisions, width, label='Precision', color='coral')
    axes[1, 0].bar(x + width/2, recalls, width, label='Recall', color='lightblue')
    axes[1, 0].set_xlabel('Models')
    axes[1, 0].set_ylabel('Score')
    axes[1, 0].set_title('Precision vs Recall')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(models, rotation=45, ha='right')
    axes[1, 0].legend()
    axes[1, 0].set_ylim([0, 1])
    
    # 4. Training time
    train_times = [r['train_time'] for r in results_list]
    axes[1, 1].barh(models, train_times, color='plum')
    axes[1, 1].set_xlabel('Training Time (seconds)')
    axes[1, 1].set_title('Training Time Comparison')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Comparison plot saved to: {output_path}")


def plot_confusion_matrices(results_list, y_test, output_path):
    """Plot confusion matrices for all models"""
    n_models = len(results_list)
    fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 4))
    
    if n_models == 1:
        axes = [axes]
    
    for idx, result in enumerate(results_list):
        cm = confusion_matrix(y_test, result['predictions'])
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Legitimate', 'Phishing'],
            yticklabels=['Legitimate', 'Phishing'],
            ax=axes[idx]
        )
        axes[idx].set_title(f"{result['model_name']}\nAccuracy: {result['accuracy']:.3f}")
        axes[idx].set_xlabel('Predicted')
        axes[idx].set_ylabel('Actual')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Confusion matrices saved to: {output_path}")


def main():
    """Main comparison workflow"""
    print("="*60)
    print("Model Comparison & Hyperparameter Tuning")
    print("="*60)
    
    # Load data
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    
    print(f"\nTrain samples: {len(X_train)}, Test samples: {len(X_test)}")
    
    # Define models to test
    models = [
        (RandomForestClassifier(n_estimators=300, max_depth=20, random_state=RANDOM_STATE, 
                               n_jobs=-1, class_weight='balanced'), 
         "Random Forest"),
        (GradientBoostingClassifier(n_estimators=200, max_depth=5, random_state=RANDOM_STATE),
         "Gradient Boosting"),
        (LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, class_weight='balanced'),
         "Logistic Regression"),
    ]
    
    if XGBOOST_AVAILABLE:
        models.append((XGBClassifier(n_estimators=300, max_depth=5, random_state=RANDOM_STATE,
                                    n_jobs=-1, eval_metric='logloss'),
                      "XGBoost"))
    
    # Evaluate all models
    results_list = []
    for model, name in models:
        result = evaluate_model(model, X_train, X_test, y_train, y_test, name)
        results_list.append(result)
    
    # Find best model
    best_result = max(results_list, key=lambda x: x['f1_score'])
    print(f"\n{'='*60}")
    print(f"🏆 BEST MODEL: {best_result['model_name']}")
    print(f"   F1-Score: {best_result['f1_score']:.4f}")
    print(f"   Accuracy: {best_result['accuracy']:.4f}")
    print(f"{'='*60}")
    
    # Save best model
    best_model_path = MODEL_DIR / "best_model.pkl"
    joblib.dump(best_result['model_object'], best_model_path)
    print(f"✅ Best model saved to: {best_model_path}")
    
    # Create visualizations
    plot_comparison(results_list, DOCS_DIR / "model_comparison.png")
    plot_confusion_matrices(results_list, y_test, DOCS_DIR / "confusion_matrices.png")
    
    # Save results summary
    summary_df = pd.DataFrame([{
        'Model': r['model_name'],
       'Accuracy': f"{r['accuracy']:.4f}",
        'Precision': f"{r['precision']:.4f}",
        'Recall': f"{r['recall']:.4f}",
        'F1-Score': f"{r['f1_score']:.4f}",
        'CV Mean': f"{r['cv_mean']:.4f}",
        'Train Time (s)': f"{r['train_time']:.2f}"
    } for r in results_list])
    
    summary_path = DOCS_DIR / "model_comparison_results.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"✅ Results summary saved to: {summary_path}")
    
    print("\n🎉 Model comparison complete!")
    print(f"📊 Check the docs/ folder for visualizations")


if __name__ == "__main__":
    main()
