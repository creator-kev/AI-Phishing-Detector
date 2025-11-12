#!/usr/bin/env python3
"""
fix_phishing_detector.py

Automated script to fix and setup the AI Phishing Detector project.
Handles all common issues and prepares the project for use.

Usage: python fix_phishing_detector.py
"""

import os
import sys
import subprocess
from pathlib import Path
import shutil

class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def print_header(msg):
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*60}")
    print(f"  {msg}")
    print(f"{'='*60}{Colors.ENDC}\n")

def print_success(msg):
    print(f"{Colors.OKGREEN}✅ {msg}{Colors.ENDC}")

def print_error(msg):
    print(f"{Colors.FAIL}❌ {msg}{Colors.ENDC}")

def print_warning(msg):
    print(f"{Colors.WARNING}⚠️  {msg}{Colors.ENDC}")

def print_info(msg):
    print(f"{Colors.OKCYAN}ℹ️  {msg}{Colors.ENDC}")

def check_python_version():
    """Ensure Python 3.7+"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 7):
        print_error(f"Python 3.7+ required. You have {version.major}.{version.minor}")
        return False
    print_success(f"Python {version.major}.{version.minor}.{version.micro}")
    return True

def check_venv():
    """Check if virtual environment is activated"""
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print_success("Virtual environment detected")
        return True
    else:
        print_warning("No virtual environment detected")
        print_info("Recommended: activate venv first")
        return True  # Don't block, just warn

def install_dependencies():
    """Install required packages"""
    print_info("Installing dependencies...")
    
    required = [
        'pandas', 'numpy', 'scikit-learn', 'flask', 
        'matplotlib', 'seaborn', 'joblib', 'colorama'
    ]
    
    try:
        subprocess.check_call([
            sys.executable, '-m', 'pip', 'install', '--quiet'
        ] + required)
        print_success("Dependencies installed")
        return True
    except subprocess.CalledProcessError:
        print_error("Failed to install dependencies")
        return False

def create_directories():
    """Ensure all required directories exist"""
    dirs = ['data', 'models', 'docs', 'notebooks', 'app']
    
    for dir_name in dirs:
        Path(dir_name).mkdir(parents=True, exist_ok=True)
    
    print_success(f"Created directories: {', '.join(dirs)}")
    return True

def generate_dataset():
    """Create sample phishing dataset"""
    print_info("Generating sample dataset...")
    
    dataset_code = """import pandas as pd

legitimate = [
    "https://www.google.com", "https://www.facebook.com",
    "https://www.amazon.com", "https://github.com",
    "https://stackoverflow.com", "https://www.wikipedia.org",
    "https://www.reddit.com", "https://www.youtube.com",
    "https://www.linkedin.com", "https://www.twitter.com",
    "https://www.microsoft.com", "https://www.apple.com",
    "https://www.netflix.com", "https://www.paypal.com",
    "https://www.instagram.com", "https://www.tiktok.com",
    "https://www.adobe.com", "https://aws.amazon.com",
    "https://www.shopify.com", "https://www.ebay.com",
    "https://www.walmart.com", "https://www.target.com",
    "https://www.bestbuy.com", "https://www.nike.com",
    "https://www.espn.com", "https://www.nytimes.com",
    "https://www.wsj.com", "https://www.forbes.com",
    "https://www.medium.com", "https://www.bbc.com",
    "https://www.cnn.com", "https://mail.google.com",
    "https://docs.google.com", "https://drive.google.com",
    "https://www.chase.com", "https://www.wellsfargo.com",
    "https://www.bankofamerica.com", "https://azure.microsoft.com",
    "https://cloud.google.com", "https://www.salesforce.com",
]

phishing = [
    "http://paypal-secure.tk/login", "http://amazon-verify.xyz/account",
    "http://192.168.1.1/admin", "https://bank-security-update.com/signin",
    "http://microsoft-account-verify.tk", "http://secure-paypal-login.tk",
    "http://apple-id-verify.com/login", "http://bank-alert.xyz/verify",
    "http://account-verify-amazon.com", "http://192.168.0.1/login",
    "http://netflix-billing-update.tk/payment", "http://facebook-security-check.xyz/verify",
    "http://google-account-recovery.tk/signin", "http://chase-bank-alert.com/secure",
    "http://wellsfargo-verify-account.xyz/login", "http://187.234.12.45/admin/login",
    "http://secure-login-paypal.tk/verify", "http://amazon-security-alert.com/update",
    "http://microsoft-support-verify.xyz/account", "http://apple-security-check.tk/signin",
    "http://account-suspended-paypal.com/restore", "http://urgent-netflix-billing.xyz/update",
    "http://verify-your-facebook.tk/security", "http://google-unusual-activity.com/verify",
    "http://bank-of-america-alert.xyz/signin", "http://203.45.67.89/secure/login",
    "http://instagram-verify-account.tk/security", "http://twitter-account-suspended.xyz/appeal",
    "http://linkedin-security-check.tk/verify", "http://dropbox-storage-full.com/upgrade",
    "http://icloud-storage-upgrade.xyz/payment", "http://spotify-premium-expired.tk/renew",
    "http://steam-account-verification.com/secure", "http://ebay-seller-verification.xyz/confirm",
    "http://walmart-gift-card-winner.tk/claim", "http://target-account-verify.com/signin",
    "http://bestbuy-order-confirmation.xyz/track", "http://fedex-package-delivery.tk/track",
    "http://ups-delivery-failed.com/reschedule", "http://dhl-customs-payment.xyz/pay",
]

data = []
for url in legitimate:
    data.append({"url": url, "class": 0})
for url in phishing:
    data.append({"url": url, "class": 1})

df = pd.DataFrame(data)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
df.to_csv("data/phishing.csv", index=False)
print(f"Created dataset: {len(df)} samples ({(df['class']==0).sum()} legitimate, {(df['class']==1).sum()} phishing)")
"""
    
    try:
        exec(dataset_code)
        print_success("Dataset created: data/phishing.csv")
        return True
    except Exception as e:
        print_error(f"Failed to create dataset: {e}")
        return False

def backup_existing_files():
    """Backup existing files before overwriting"""
    files_to_backup = [
        'notebooks/train_pipeline.py',
        'app/main.py',
        'notebooks/evaluate_model.py'
    ]
    
    backed_up = []
    for file_path in files_to_backup:
        if os.path.exists(file_path):
            backup_path = f"{file_path}.backup"
            shutil.copy2(file_path, backup_path)
            backed_up.append(file_path)
    
    if backed_up:
        print_success(f"Backed up {len(backed_up)} files (.backup extension)")
    
    return True

def check_model_exists():
    """Check if trained model exists"""
    model_path = Path("models/phishing_pipeline.pkl")
    
    if model_path.exists():
        size = model_path.stat().st_size / 1024  # KB
        print_success(f"Model exists: {size:.1f} KB")
        return True
    else:
        print_warning("Model not found - needs training")
        return False

def run_training():
    """Execute training pipeline"""
    print_info("Starting model training...")
    
    try:
        result = subprocess.run(
            [sys.executable, "notebooks/train_pipeline.py"],
            capture_output=True,
            text=True,
            timeout=120
        )
        
        if result.returncode == 0:
            print_success("Training completed successfully")
            print_info("Check output above for accuracy metrics")
            return True
        else:
            print_error("Training failed")
            print(result.stderr)
            return False
    
    except subprocess.TimeoutExpired:
        print_error("Training timed out (>2 minutes)")
        return False
    except FileNotFoundError:
        print_error("train_pipeline.py not found")
        return False
    except Exception as e:
        print_error(f"Training error: {e}")
        return False

def verify_installation():
    """Run final verification checks"""
    print_info("Running verification checks...")
    
    checks = {
        "Dataset exists": Path("data/phishing.csv").exists(),
        "Model exists": Path("models/phishing_pipeline.pkl").exists(),
        "Flask app exists": Path("app/main.py").exists(),
        "Training script exists": Path("notebooks/train_pipeline.py").exists(),
    }
    
    all_passed = all(checks.values())
    
    for check, passed in checks.items():
        if passed:
            print_success(check)
        else:
            print_error(check)
    
    return all_passed

def print_next_steps():
    """Show user what to do next"""
    print_header("🎉 Setup Complete!")
    
    print(f"{Colors.BOLD}Next Steps:{Colors.ENDC}\n")
    
    steps = [
        ("1️⃣", "Start Flask app", "python app/main.py"),
        ("2️⃣", "Open browser", "http://127.0.0.1:5000"),
        ("3️⃣", "Run tests", "python test_phishing_detector.py"),
        ("4️⃣", "Check evaluation", "python notebooks/evaluate_model.py"),
    ]
    
    for icon, desc, cmd in steps:
        print(f"{icon}  {Colors.OKGREEN}{desc}{Colors.ENDC}")
        print(f"   {Colors.OKCYAN}$ {cmd}{Colors.ENDC}\n")
    
    print(f"{Colors.BOLD}Test URLs:{Colors.ENDC}")
    print(f"  {Colors.OKGREEN}✅ Legitimate: https://www.google.com{Colors.ENDC}")
    print(f"  {Colors.FAIL}⚠️  Phishing: http://paypal-secure.tk/login{Colors.ENDC}\n")

def main():
    """Main setup workflow"""
    print_header("🛡️  AI Phishing Detector - Automated Setup")
    
    steps = [
        ("Checking Python version", check_python_version),
        ("Checking virtual environment", check_venv),
        ("Creating directories", create_directories),
        ("Installing dependencies", install_dependencies),
        ("Generating dataset", generate_dataset),
        ("Backing up existing files", backup_existing_files),
    ]
    
    for desc, func in steps:
        print_info(f"{desc}...")
        if not func():
            print_error(f"Failed: {desc}")
            sys.exit(1)
    
    # Check if model needs training
    if not check_model_exists():
        print_warning("Model not found. Training required.")
        response = input(f"{Colors.OKCYAN}Train model now? (y/n): {Colors.ENDC}")
        
        if response.lower() in ['y', 'yes']:
            if not run_training():
                print_warning("Training failed. Please run manually:")
                print(f"{Colors.OKCYAN}python notebooks/train_pipeline.py{Colors.ENDC}")
    
    # Final verification
    print_header("Verification")
    if verify_installation():
        print_next_steps()
    else:
        print_error("Some checks failed. Review errors above.")
        sys.exit(1)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n{Colors.WARNING}Setup cancelled by user{Colors.ENDC}")
        sys.exit(1)
    except Exception as e:
        print_error(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
