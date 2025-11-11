# exploration.py
import os
import sys
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DATA_PATH = os.path.join(os.path.dirname(__file__), "data/phishing.csv")

def main():
    if not os.path.exists(DATA_PATH):
        print(f"ERROR: dataset not found at: {DATA_PATH}", file=sys.stderr)
        sys.exit(2)
    
    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.strip().str.lower()
    
    print("="*60)
    print("Dataset Exploration")
    print("="*60)
    print(f"Loaded: {DATA_PATH}")
    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    print("\nFirst 5 rows:")
    print(df.head(5))
    
    print("\nMissing values:")
    print(df.isnull().sum())
    
    # Find label column
    label_col = None
    for col in ["class", "label", "result", "target"]:
        if col in df.columns:
            label_col = col
            break
    
    if label_col:
        counts = df[label_col].value_counts()
        print(f"\nLabel distribution ({label_col}):")
        print(counts)
        
        # Plot
        ax = counts.plot(kind="bar")
        ax.set_xlabel(label_col)
        ax.set_ylabel("Count")
        plt.title("Label Distribution")
        plt.tight_layout()
        
        out_path = os.path.join(os.path.dirname(__file__), "docs/label_distribution.png")
        plt.savefig(out_path)
        print(f"\n✅ Saved plot to: {out_path}")

if __name__ == "__main__":
    main()
EOF

# Create sample dataset
echo "[9/10] Creating sample dataset (data/phishing.csv)..."
cat > data/phishing.csv << 'EOF'
url,class
https://www.google.com,0
https://www.facebook.com,0
https://www.amazon.com,0
https://github.com,0
https://stackoverflow.com,0
http://paypal-secure.tk/login,1
http://amazon-verify.xyz/account,1
http://192.168.1.1/admin,1
https://bank-security-update.com/signin,1
http://microsoft-account-verify.tk,1
https://www.wikipedia.org,0
https://www.reddit.com,0
http://secure-paypal-login.tk,1
http://apple-id-verify.com/login,1
https://www.youtube.com,0
http://192.168.0.1/login,1
https://www.linkedin.com,0
http://bank-alert.xyz/verify,1
https://www.twitter.com,0
http://account-verify-amazon.com,1
EOF

