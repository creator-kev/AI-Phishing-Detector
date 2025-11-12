# notebooks/exploration.py
"""
AI Phishing Detector - Data Exploration Script
==============================================
Comprehensive exploratory data analysis including:
- Dataset overview and statistics
- Label distribution visualization
- Feature analysis
- URL pattern analysis
- Missing values detection
- Data quality checks
- Correlation analysis (for feature-based datasets)
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
from datetime import datetime

# -----------------------
# Configuration
# -----------------------
SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parent.parent # always go up one level from notebooks -> repo root
DATA_PATH = PROJECT_ROOT / "data" / "phishing.csv"
DOCS_PATH = PROJECT_ROOT / "docs"

# Create docs directory
DOCS_PATH.mkdir(parents=True, exist_ok=True)

# Styling
sns.set_style("whitegrid")
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

print("\n" + "="*70)
print("🔍 AI PHISHING DETECTOR - DATA EXPLORATION")
print("="*70)
print(f"📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"📁 Project Root: {PROJECT_ROOT}")
print(f"📊 Data Path: {DATA_PATH}")
print("="*70 + "\n")

# -----------------------
# Load Dataset
# -----------------------
print("📂 Loading dataset...")

if not DATA_PATH.exists():
    print(f"❌ ERROR: Dataset not found at {DATA_PATH}")
    print("\n💡 To create a sample dataset, run:")
    print("   python generate_sample_dataset.py")
    sys.exit(1)

df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip()

print(f"✅ Dataset loaded successfully\n")

# -----------------------
# Basic Information
# -----------------------
print("="*70)
print("📋 BASIC DATASET INFORMATION")
print("="*70 + "\n")

print(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"Memory Usage: {df.memory_usage(deep=True).sum() / 1024:.2f} KB\n")

print("Columns:")
for i, col in enumerate(df.columns, 1):
    dtype = df[col].dtype
    non_null = df[col].notna().sum()
    print(f"  {i}. {col:20s} | Type: {str(dtype):10s} | Non-null: {non_null}/{len(df)}")

print()

# -----------------------
# First Few Rows
# -----------------------
print("="*70)
print("👀 FIRST 10 ROWS")
print("="*70)
print(df.head(10).to_string())
print()

# -----------------------
# Statistical Summary
# -----------------------
print("="*70)
print("📊 STATISTICAL SUMMARY")
print("="*70)
print(df.describe(include='all').to_string())
print()

# -----------------------
# Missing Values
# -----------------------
print("="*70)
print("🔍 MISSING VALUES ANALYSIS")
print("="*70 + "\n")

missing = df.isnull().sum()
if missing.sum() == 0:
    print("✅ No missing values detected!")
else:
    print("Missing values per column:")
    for col in missing[missing > 0].index:
        count = missing[col]
        percent = (count / len(df)) * 100
        print(f"  • {col}: {count} ({percent:.2f}%)")

print()

# -----------------------
# Find Label Column
# -----------------------
label_col = None
for col in ["class", "Class", "label", "Label", "result", "Result", "target"]:
    if col in df.columns:
        label_col = col
        break

if label_col is None:
    print("⚠️  No label column found")
    label_col = df.columns[-1]  # Assume last column
    print(f"Assuming '{label_col}' is the label column\n")
else:
    print(f"🏷️  Label column: '{label_col}'\n")

# -----------------------
# Label Distribution
# -----------------------
print("="*70)
print("📊 LABEL DISTRIBUTION")
print("="*70 + "\n")

label_counts = df[label_col].value_counts().sort_index()
print("Label counts:")
for label, count in label_counts.items():
    percent = (count / len(df)) * 100
    label_name = "Phishing" if label in [1, "1", "phishing"] else "Legitimate"
    print(f"  {label} ({label_name}): {count:4d} ({percent:.2f}%)")

print()

# Check for class imbalance
ratio = label_counts.max() / label_counts.min() if len(label_counts) > 1 else 1.0
if ratio > 2:
    print(f"⚠️  Class imbalance detected! Ratio: {ratio:.2f}:1")
    print("   Consider using class_weight='balanced' in training")
elif ratio > 1.5:
    print(f"⚠️  Slight class imbalance. Ratio: {ratio:.2f}:1")
else:
    print(f"✅ Well-balanced dataset. Ratio: {ratio:.2f}:1")

print()

# -----------------------
# Visualization 1: Label Distribution
# -----------------------
print("="*70)
print("🎨 GENERATING VISUALIZATIONS")
print("="*70 + "\n")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Bar chart
colors = ['#48bb78', '#fc8181']  # Green for legitimate, red for phishing
label_names = []
for label in label_counts.index:
    if label in [1, "1", "phishing", "Phishing"]:
        label_names.append("Phishing")
    else:
        label_names.append("Legitimate")

bars = ax1.bar(label_names, label_counts.values, color=colors, alpha=0.8, 
               edgecolor='black', linewidth=2)
ax1.set_ylabel('Count', fontsize=12, weight='bold')
ax1.set_title('Label Distribution', fontsize=14, weight='bold', pad=20)
ax1.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bar, value in zip(bars, label_counts.values):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + len(df)*0.01,
             f'{value}\n({value/len(df)*100:.1f}%)',
             ha='center', va='bottom', fontsize=11, weight='bold')

# Pie chart
ax2.pie(label_counts.values, labels=label_names, autopct='%1.1f%%',
        colors=colors, startangle=90, textprops={'fontsize': 12, 'weight': 'bold'},
        explode=[0.05, 0.05])
ax2.set_title('Label Distribution (Percentage)', fontsize=14, weight='bold', pad=20)

plt.tight_layout()
dist_path = DOCS_PATH / "label_distribution.png"
plt.savefig(dist_path, dpi=300, bbox_inches='tight')
print(f"✅ Label distribution saved: {dist_path}")
plt.close()

# -----------------------
# URL Analysis (if URL column exists)
# -----------------------
has_url = any(c.lower() == "url" for c in df.columns)

if has_url:
    url_col = next(c for c in df.columns if c.lower() == "url")
    
    print("\n" + "="*70)
    print("🌐 URL PATTERN ANALYSIS")
    print("="*70 + "\n")
    
    urls = df[url_col].astype(str)
    
    # URL lengths
    url_lengths = urls.str.len()
    print(f"URL Length Statistics:")
    print(f"  Min:    {url_lengths.min()}")
    print(f"  Max:    {url_lengths.max()}")
    print(f"  Mean:   {url_lengths.mean():.2f}")
    print(f"  Median: {url_lengths.median():.0f}\n")
    
    # Protocol analysis
    protocols = urls.str.extract(r'^(https?):')[0].value_counts()
    print("Protocol Distribution:")
    for proto, count in protocols.items():
        print(f"  {proto:5s}: {count:4d} ({count/len(urls)*100:.1f}%)")
    print()
    
    # TLD analysis
    tlds = urls.str.extract(r'\.([a-z]{2,})(?:/|$)', flags=re.IGNORECASE)[0]
    tld_counts = tlds.value_counts().head(10)
    print("Top 10 TLDs:")
    for tld, count in tld_counts.items():
        print(f"  .{tld:10s}: {count:4d}")
    print()
    
    # IP address detection
    has_ip = urls.str.contains(r'\d+\.\d+\.\d+\.\d+', regex=True).sum()
    print(f"URLs with IP addresses: {has_ip} ({has_ip/len(urls)*100:.2f}%)")
    
    # Suspicious keywords
    suspicious_keywords = ['login', 'verify', 'secure', 'account', 'bank', 'update', 'confirm']
    keyword_counts = {}
    for keyword in suspicious_keywords:
        count = urls.str.lower().str.contains(keyword).sum()
        if count > 0:
            keyword_counts[keyword] = count
    
    if keyword_counts:
        print(f"\nSuspicious Keywords Found:")
        for keyword, count in sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  '{keyword}': {count} URLs")
    print()
    
    # -----------------------
    # Visualization 2: URL Length Distribution
    # -----------------------
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # URL length by class
    for label in df[label_col].unique():
        mask = df[label_col] == label
        label_name = "Phishing" if label in [1, "1"] else "Legitimate"
        color = '#fc8181' if label in [1, "1"] else '#48bb78'
        
        axes[0, 0].hist(url_lengths[mask], bins=30, alpha=0.6, 
                       label=label_name, color=color, edgecolor='black')
    
    axes[0, 0].set_xlabel('URL Length (characters)', fontsize=11, weight='bold')
    axes[0, 0].set_ylabel('Frequency', fontsize=11, weight='bold')
    axes[0, 0].set_title('URL Length Distribution by Class', fontsize=12, weight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    
    # Box plot of URL length
    data_box = pd.DataFrame({
        'Length': url_lengths,
        'Class': [('Phishing' if x in [1, "1"] else 'Legitimate') for x in df[label_col]]
    })
    sns.boxplot(x='Class', y='Length', data=data_box, ax=axes[0, 1], 
                palette=['#48bb78', '#fc8181'])
    axes[0, 1].set_title('URL Length by Class', fontsize=12, weight='bold')
    axes[0, 1].set_ylabel('Length (characters)', fontsize=11, weight='bold')
    axes[0, 1].grid(axis='y', alpha=0.3)
    
    # Protocol distribution
    protocol_data = urls.str.extract(r'^(https?):')[0].fillna('other')
    protocol_by_class = pd.crosstab(df[label_col], protocol_data)
    protocol_by_class.index = ['Legitimate' if x not in [1, "1"] else 'Phishing' 
                               for x in protocol_by_class.index]
    protocol_by_class.plot(kind='bar', ax=axes[1, 0], color=['#4299e1', '#ed8936'], 
                           alpha=0.8, edgecolor='black', linewidth=1.5)
    axes[1, 0].set_title('Protocol Usage by Class', fontsize=12, weight='bold')
    axes[1, 0].set_ylabel('Count', fontsize=11, weight='bold')
    axes[1, 0].set_xlabel('Class', fontsize=11, weight='bold')
    axes[1, 0].legend(title='Protocol')
    axes[1, 0].tick_params(axis='x', rotation=0)
    axes[1, 0].grid(axis='y', alpha=0.3)
    
    # TLD distribution (top 10)
    if not tld_counts.empty:
        top_tlds = tld_counts.head(10)
        axes[1, 1].barh(range(len(top_tlds)), top_tlds.values, 
                       color=plt.cm.viridis(np.linspace(0.3, 0.9, len(top_tlds))),
                       edgecolor='black', linewidth=1)
        axes[1, 1].set_yticks(range(len(top_tlds)))
        axes[1, 1].set_yticklabels([f'.{tld}' for tld in top_tlds.index])
        axes[1, 1].set_xlabel('Count', fontsize=11, weight='bold')
        axes[1, 1].set_title('Top 10 TLDs', fontsize=12, weight='bold')
        axes[1, 1].invert_yaxis()
        axes[1, 1].grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    url_analysis_path = DOCS_PATH / "url_analysis.png"
    plt.savefig(url_analysis_path, dpi=300, bbox_inches='tight')
    print(f"✅ URL analysis saved: {url_analysis_path}")
    plt.close()

# -----------------------
# Feature Analysis (if features exist)
# -----------------------
feature_cols = [c for c in df.columns if c not in [label_col, url_col if has_url else None, 'index', 'Index', 'Unnamed: 0']]
feature_cols = [c for c in feature_cols if c is not None]

if feature_cols and len(feature_cols) > 0:
    print("\n" + "="*70)
    print("📊 FEATURE ANALYSIS")
    print("="*70 + "\n")
    
    print(f"Number of features: {len(feature_cols)}")
    print(f"Features: {feature_cols[:10]}{'...' if len(feature_cols) > 10 else ''}\n")
    
    # Convert to numeric
    feature_df = df[feature_cols].copy()
    for col in feature_df.columns:
        feature_df[col] = pd.to_numeric(feature_df[col], errors='coerce')
    
    # Feature statistics by class
    print("Feature Statistics by Class:")
    print("-"*70)
    
    for col in feature_cols[:5]:  # Show first 5 features
        print(f"\n{col}:")
        for label in df[label_col].unique():
            mask = df[label_col] == label
            label_name = "Phishing" if label in [1, "1"] else "Legitimate"
            values = feature_df.loc[mask, col]
            print(f"  {label_name:12s}: Mean={values.mean():.3f}, Std={values.std():.3f}, "
                  f"Min={values.min():.3f}, Max={values.max():.3f}")
    
    if len(feature_cols) > 5:
        print(f"\n... and {len(feature_cols) - 5} more features")
    print()
    
    # -----------------------
    # Visualization 3: Feature Correlation Heatmap
    # -----------------------
    if len(feature_cols) >= 2:
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Calculate correlation
        corr_df = feature_df.copy()
        corr_df[label_col] = df[label_col].replace({-1: 0})  # Normalize labels
        corr_matrix = corr_df.corr()
        
        # Plot heatmap
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=len(feature_cols) <= 15, 
                   fmt='.2f', cmap='coolwarm', center=0, 
                   square=True, linewidths=1, cbar_kws={"shrink": 0.8})
        plt.title('Feature Correlation Matrix', fontsize=14, weight='bold', pad=20)
        plt.tight_layout()
        
        corr_path = DOCS_PATH / "feature_correlation.png"
        plt.savefig(corr_path, dpi=300, bbox_inches='tight')
        print(f"✅ Correlation matrix saved: {corr_path}")
        plt.close()
    
    # -----------------------
    # Visualization 4: Feature Distributions
    # -----------------------
    if len(feature_cols) >= 4:
        n_features = min(9, len(feature_cols))
        n_rows = int(np.ceil(n_features / 3))
        
        fig, axes = plt.subplots(n_rows, 3, figsize=(15, n_rows * 4))
        axes = axes.flatten() if n_features > 1 else [axes]
        
        for idx, col in enumerate(feature_cols[:n_features]):
            ax = axes[idx]
            
            for label in df[label_col].unique():
                mask = df[label_col] == label
                label_name = "Phishing" if label in [1, "1"] else "Legitimate"
                color = '#fc8181' if label in [1, "1"] else '#48bb78'
                
                values = feature_df.loc[mask, col].dropna()
                if len(values) > 0:
                    ax.hist(values, bins=20, alpha=0.6, label=label_name, 
                           color=color, edgecolor='black')
            
            ax.set_xlabel(col, fontsize=10, weight='bold')
            ax.set_ylabel('Frequency', fontsize=10, weight='bold')
            ax.set_title(f'{col} Distribution', fontsize=11, weight='bold')
            ax.legend()
            ax.grid(alpha=0.3)
        
        # Hide extra subplots
        for idx in range(n_features, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        feat_dist_path = DOCS_PATH / "feature_distributions.png"
        plt.savefig(feat_dist_path, dpi=300, bbox_inches='tight')
        print(f"✅ Feature distributions saved: {feat_dist_path}")
        plt.close()

# -----------------------
# Data Quality Report
# -----------------------
print("\n" + "="*70)
print("✅ DATA QUALITY REPORT")
print("="*70 + "\n")

quality_checks = {
    "Total samples": len(df),
    "Complete cases (no missing)": df.notna().all(axis=1).sum(),
    "Duplicate rows": df.duplicated().sum(),
    "Minimum samples per class": label_counts.min(),
    "Class balance ratio": f"{ratio:.2f}:1"
}

for check, value in quality_checks.items():
    print(f"  • {check:30s}: {value}")

print()

# Recommendations
print("📋 RECOMMENDATIONS:")
if len(df) < 100:
    print("  ⚠️  Small dataset (<100 samples). Consider collecting more data.")
if ratio > 2:
    print("  ⚠️  Class imbalance detected. Use balanced class weights during training.")
if df.duplicated().sum() > 0:
    print(f"  ⚠️  {df.duplicated().sum()} duplicate rows found. Consider removing.")
if df.isnull().sum().sum() > 0:
    print("  ⚠️  Missing values detected. Handle before training.")

if len(df) >= 100 and ratio <= 1.5 and df.duplicated().sum() == 0 and df.isnull().sum().sum() == 0:
    print("  ✅ Dataset quality looks good! Ready for training.")

print()

# -----------------------
# Save Exploration Report
# -----------------------
print("="*70)
print("💾 SAVING EXPLORATION REPORT")
print("="*70 + "\n")

report_path = DOCS_PATH / "exploration_report.txt"
with open(report_path, 'w', encoding='utf-8') as f:
    f.write("AI PHISHING DETECTOR - DATA EXPLORATION REPORT\n")
    f.write("="*70 + "\n")
    f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Dataset: {DATA_PATH}\n\n")
    
    f.write("DATASET OVERVIEW\n")
    f.write("-"*70 + "\n")
    f.write(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns\n")
    f.write(f"Memory: {df.memory_usage(deep=True).sum() / 1024:.2f} KB\n\n")
    
    f.write("COLUMNS\n")
    f.write("-"*70 + "\n")
    for i, col in enumerate(df.columns, 1):
        f.write(f"{i}. {col} ({df[col].dtype})\n")
    f.write("\n")
    
    f.write("LABEL DISTRIBUTION\n")
    f.write("-"*70 + "\n")
    for label, count in label_counts.items():
        percent = (count / len(df)) * 100
        label_name = "Phishing" if label in [1, "1"] else "Legitimate"
        f.write(f"{label} ({label_name}): {count} ({percent:.2f}%)\n")
    f.write(f"\nClass Balance Ratio: {ratio:.2f}:1\n\n")
    
    f.write("DATA QUALITY\n")
    f.write("-"*70 + "\n")
    for check, value in quality_checks.items():
        f.write(f"{check}: {value}\n")
    f.write("\n")
    
    if has_url:
        f.write("URL STATISTICS\n")
        f.write("-"*70 + "\n")
        f.write(f"Min length: {url_lengths.min()}\n")
        f.write(f"Max length: {url_lengths.max()}\n")
        f.write(f"Mean length: {url_lengths.mean():.2f}\n")
        f.write(f"URLs with IP: {has_ip}\n\n")

print(f"✅ Exploration report saved: {report_path}")

# -----------------------
# Summary
# -----------------------
print("\n" + "="*70)
print("📊 EXPLORATION SUMMARY")
print("="*70)
print(f"\n✅ Dataset: {len(df)} samples, {len(df.columns)} columns")
print(f"✅ Visualizations: {len(list(DOCS_PATH.glob('*.png')))} files")
print(f"✅ Report: {report_path}")
print(f"\n📁 All files saved in: {DOCS_PATH}")
print("\n" + "="*70)
print("🎉 EXPLORATION COMPLETE!")
print("="*70 + "\n")
