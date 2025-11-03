# AI-Phishing-Detector

Simple AI/ML project to detect phishing URLs/emails.

## Structure
- data/          -> dataset files (do not commit large raw files)
- notebooks/     -> Jupyter notebooks for EDA and modeling
- app/           -> Flask/Streamlit app for inference
- models/        -> saved trained models (.pkl)
- docs/          -> documentation and reports

## Quickstart (Kali)
1. python3 -m venv venv
2. source venv/bin/activate
3. pip install -r requirements.txt
4. jupyter notebook (open notebooks/exploration.ipynb)

## Dataset
- Source: [Kaggle - Phishing Website Detector](https://www.kaggle.com/datasets/eswarchandt/phishing-website-detector)
- Saved locally under `data/phishing.csv`
- Label column: `Result` (1 = phishing, -1 = legitimate)
