# exploration.py - quick dataset load & basic stats
import os
import sys
import pandas as pd
import matplotlib
# Use a non-interactive backend so it works on headless systems (CI, servers, Kali without X)
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# change filename if yours differs
DATA_PATH = os.path.join(os.path.dirname(__file__), "../data/phishing.csv")

def main():
    # normalize path
    data_path = os.path.abspath(DATA_PATH)
    if not os.path.exists(data_path):
        print(f"ERROR: dataset not found at: {data_path}", file=sys.stderr)
        print("Files in ../data:", file=sys.stderr)
        for f in sorted(os.listdir(os.path.join(os.path.dirname(__file__), "../data"))):
            print(" -", f, file=sys.stderr)
        sys.exit(2)

    df = pd.read_csv(data_path)
    print("Loaded:", data_path)
    print("Shape:", df.shape)
    print("Columns:", df.columns.tolist())
    print("\nSample rows:")
    print(df.head(5).to_string(index=False))
    print("\nMissing values per column:")
    print(df.isnull().sum())

    # simple class distribution plot (change 'Result' to your label column)
    label_col = "Result" if "Result" in df.columns else df.columns[-1]
    counts = df[label_col].value_counts()
    print("\nLabel distribution:")
    print(counts.to_string())

    ax = counts.plot(kind="bar")
    ax.set_xlabel(label_col)
    ax.set_ylabel("Count")
    plt.title("Label distribution")
    plt.tight_layout()

    out_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../docs/label_distribution.png"))
    plt.savefig(out_path)
    print(f"Saved label distribution plot to: {out_path}")

if __name__ == "__main__":
    main()
