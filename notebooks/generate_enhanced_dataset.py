# notebooks/generate_enhanced_dataset.py
"""
Generate an enhanced phishing dataset with realistic URLs.
This creates a much larger and more diverse dataset for training.
"""

import pandas as pd
import random
import string
from pathlib import Path

# Legitimate domains (trusted sites)
LEGITIMATE_DOMAINS = [
    "google.com", "facebook.com", "amazon.com", "microsoft.com", "apple.com",
    "github.com", "stackoverflow.com", "wikipedia.org", "reddit.com", "twitter.com",
    "linkedin.com", "youtube.com", "instagram.com", "netflix.com", "ebay.com",
    "paypal.com", "bankofamerica.com", "chase.com", "wellsfargo.com", "citi.com",
    "dropbox.com", "adobe.com", "salesforce.com", "zoom.us", "spotify.com",
    "twitch.tv", "medium.com", "wordpress.com", "shopify.com", "stripe.com",
    "gmail.com", "outlook.com", "yahoo.com", "protonmail.com", "icloud.com"
]

# Common legitimate paths
LEGITIMATE_PATHS = [
    "", "/", "/about", "/contact", "/products", "/services", "/blog",
    "/help", "/support", "/docs", "/api", "/download", "/pricing",
    "/features", "/company", "/careers", "/privacy", "/terms"
]

# Suspicious TLDs commonly used in phishing
SUSPICIOUS_TLDS = [
    ".tk", ".ml", ".ga", ".cf", ".gq", ".xyz", ".top", ".work",
    ".date", ".racing", ".win", ".bid", ".loan", ".download"
]

# Phishing keywords
PHISHING_KEYWORDS = [
    "secure", "verify", "account", "login", "signin", "update", "confirm",
    "banking", "payment", "suspended", "locked", "unusual", "activity",
    "restore", "validate", "urgent", "alert", "notification", "security"
]

# Common brands targeted in phishing
TARGETED_BRANDS = [
    "paypal", "amazon", "apple", "microsoft", "google", "facebook",
    "netflix", "instagram", "wells-fargo", "chase", "bankofamerica",
    "dhl", "fedex", "usps", "irs", "irs-gov"
]

def generate_legitimate_url():
    """Generate a realistic legitimate URL"""
    domain = random.choice(LEGITIMATE_DOMAINS)
    protocol = random.choice(["https://", "http://"])
    
    # 90% chance of https for legitimate sites
    if random.random() < 0.9:
        protocol = "https://"
    
    # Sometimes add www
    if random.random() < 0.4:
        domain = "www." + domain
    
    # Add path
    path = random.choice(LEGITIMATE_PATHS)
    
    # Sometimes add query parameters
    if random.random() < 0.2:
        params = ["?page=1", "?id=123", "?sort=new", "?filter=active"]
        path += random.choice(params)
    
    return protocol + domain + path

def generate_phishing_url():
    """Generate a realistic phishing URL"""
    templates = [
        generate_typosquatting_url,
        generate_subdomain_phishing_url,
        generate_ip_phishing_url,
        generate_keyword_stuffed_url,
        generate_suspicious_tld_url,
        generate_url_shortener_mimic,
        generate_homograph_url
    ]
    
    return random.choice(templates)()

def generate_typosquatting_url():
    """Generate URLs with typos in domain names"""
    brand = random.choice(TARGETED_BRANDS)
    typos = [
        brand.replace('a', '4'), brand.replace('o', '0'), 
        brand.replace('e', '3'), brand + 's',
        brand.replace('l', '1'), brand.replace('i', '1'),
        brand + 'secure', brand + 'login'
    ]
    domain = random.choice(typos) + random.choice(SUSPICIOUS_TLDS)
    keyword = random.choice(PHISHING_KEYWORDS)
    return f"http://{domain}/{keyword}"

def generate_subdomain_phishing_url():
    """Generate URLs using subdomains to deceive"""
    brand = random.choice(TARGETED_BRANDS)
    keyword = random.choice(PHISHING_KEYWORDS)
    suspicious_domain = "".join(random.choices(string.ascii_lowercase, k=8))
    tld = random.choice(SUSPICIOUS_TLDS + [".com", ".net", ".org"])
    
    patterns = [
        f"http://{brand}-{keyword}.{suspicious_domain}{tld}",
        f"http://{keyword}.{brand}.{suspicious_domain}{tld}",
        f"http://{brand}.{keyword}-{suspicious_domain}{tld}",
        f"https://{brand}-verify.{suspicious_domain}{tld}/login"
    ]
    return random.choice(patterns)

def generate_ip_phishing_url():
    """Generate URLs using IP addresses"""
    ip = f"{random.randint(1,255)}.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(1,255)}"
    keyword = random.choice(PHISHING_KEYWORDS)
    return f"http://{ip}/{keyword}"

def generate_keyword_stuffed_url():
    """Generate URLs stuffed with phishing keywords"""
    brand = random.choice(TARGETED_BRANDS)
    keywords = random.sample(PHISHING_KEYWORDS, 2)
    domain = "-".join([brand] + keywords) + random.choice([".com", ".net"] + SUSPICIOUS_TLDS)
    return f"http://{domain}/login"

def generate_suspicious_tld_url():
    """Generate URLs with suspicious TLDs"""
    brand = random.choice(TARGETED_BRANDS)
    tld = random.choice(SUSPICIOUS_TLDS)
    keyword = random.choice(PHISHING_KEYWORDS)
    return f"http://{brand}-{keyword}{tld}/account"

def generate_url_shortener_mimic():
    """Generate URLs mimicking URL shorteners"""
    chars = ''.join(random.choices(string.ascii_letters + string.digits, k=6))
    domains = ["bit-ly.tk", "tinyurl.xyz", "goo-gl.ml", "short.cf"]
    return f"http://{random.choice(domains)}/{chars}"

def generate_homograph_url():
    """Generate URLs using similar-looking characters"""
    # Simple version - replace with similar chars
    brand = random.choice(TARGETED_BRANDS)
    # Replace with visually similar characters
    brand = brand.replace('o', '0').replace('l', '1').replace('i', '1')
    return f"http://{brand}{random.choice(SUSPICIOUS_TLDS)}/signin"

def generate_dataset(n_samples=5000, phishing_ratio=0.5):
    """
    Generate a balanced dataset.
    
    Args:
        n_samples: Total number of URLs to generate
        phishing_ratio: Proportion of phishing URLs (0.5 = balanced)
    """
    n_phishing = int(n_samples * phishing_ratio)
    n_legitimate = n_samples - n_phishing
    
    print(f"Generating {n_samples} URLs...")
    print(f"  - Legitimate: {n_legitimate}")
    print(f"  - Phishing: {n_phishing}")
    
    urls = []
    labels = []
    
    # Generate legitimate URLs
    for _ in range(n_legitimate):
        urls.append(generate_legitimate_url())
        labels.append(0)  # 0 = legitimate
    
    # Generate phishing URLs
    for _ in range(n_phishing):
        urls.append(generate_phishing_url())
        labels.append(1)  # 1 = phishing
    
    # Create DataFrame and shuffle
    df = pd.DataFrame({'url': urls, 'class': labels})
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    return df

def main():
    """Generate and save enhanced dataset"""
    ROOT = Path(__file__).resolve().parents[1]
    DATA_DIR = ROOT / "data"
    DATA_DIR.mkdir(exist_ok=True)
    
    # Generate dataset
    print("="*60)
    print("Enhanced Dataset Generation")
    print("="*60)
    
    df = generate_dataset(n_samples=10000, phishing_ratio=0.5)
    
    # Save to CSV
    output_path = DATA_DIR / "phishing_enhanced.csv"
    df.to_csv(output_path, index=False)
    
    print(f"\n✅ Dataset saved to: {output_path}")
    print(f"   Total URLs: {len(df)}")
    print(f"   Legitimate: {(df['class'] == 0).sum()}")
    print(f"   Phishing: {(df['class'] == 1).sum()}")
    print("\nSample URLs:")
    print(df.head(10))
    
    # Show distribution
    print("\nClass Distribution:")
    print(df['class'].value_counts())

if __name__ == "__main__":
    main()
