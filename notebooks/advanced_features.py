# notebooks/advanced_features.py
"""
Advanced feature extraction for phishing detection.
Extracts comprehensive features from URLs for better model performance.
"""

import re
import urllib.parse
from urllib.parse import urlparse
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from pathlib import Path

class AdvancedURLFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Extract 30+ advanced features from URLs for phishing detection.
    """
    
    def __init__(self):
        # Suspicious keywords
        self.phishing_keywords = [
            'login', 'signin', 'secure', 'account', 'update', 'verify',
            'confirm', 'banking', 'suspended', 'locked', 'urgent', 'alert',
            'validate', 'restore', 'security', 'notification'
        ]
        
        # Trusted TLDs
        self.trusted_tlds = ['.com', '.org', '.net', '.edu', '.gov', '.mil']
        
        # Suspicious TLDs
        self.suspicious_tlds = [
            '.tk', '.ml', '.ga', '.cf', '.gq', '.xyz', '.top', '.work',
            '.date', '.racing', '.win', '.bid', '.loan', '.download'
        ]
        
        # Common brands targeted in phishing
        self.targeted_brands = [
            'paypal', 'amazon', 'apple', 'microsoft', 'google', 'facebook',
            'netflix', 'instagram', 'wellsfargo', 'chase', 'bankofamerica'
        ]
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        """Extract features from URLs"""
        if isinstance(X, pd.Series):
            X = X.values
        elif isinstance(X, pd.DataFrame):
            X = X.values.ravel()
        
        features_list = []
        for url in X:
            features_list.append(self._extract_features(str(url)))
        
        return np.array(features_list)
    
    def _extract_features(self, url):
        """Extract all features from a single URL"""
        features = {}
        url = url.strip().lower()
        
        # Parse URL
        try:
            parsed = urlparse(url)
            domain = parsed.netloc
            path = parsed.path
            query = parsed.query
        except:
            parsed = None
            domain = ''
            path = ''
            query = ''
        
        # === BASIC FEATURES ===
        features['url_length'] = len(url)
        features['domain_length'] = len(domain)
        features['path_length'] = len(path)
        features['query_length'] = len(query)
        
        # === CHARACTER COUNTS ===
        features['num_dots'] = url.count('.')
        features['num_hyphens'] = url.count('-')
        features['num_underscores'] = url.count('_')
        features['num_slashes'] = url.count('/')
        features['num_questionmarks'] = url.count('?')
        features['num_equals'] = url.count('=')
        features['num_at'] = url.count('@')
        features['num_ampersands'] = url.count('&')
        features['num_percent'] = url.count('%')
        features['num_digits'] = sum(c.isdigit() for c in url)
        
        # === SPECIAL CHARACTER RATIOS ===
        if len(url) > 0:
            features['digit_ratio'] = features['num_digits'] / len(url)
            features['special_char_ratio'] = sum(not c.isalnum() for c in url) / len(url)
        else:
            features['digit_ratio'] = 0
            features['special_char_ratio'] = 0
        
        # === PROTOCOL FEATURES ===
        features['has_https'] = int(url.startswith('https://'))
        features['has_http'] = int(url.startswith('http://'))
        features['no_protocol'] = int(not url.startswith('http'))
        
        # === SUSPICIOUS PATTERNS ===
        features['has_ip'] = int(bool(re.search(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', url)))
        features['has_at_symbol'] = int('@' in url)
        features['has_double_slash'] = int(url.count('//') > 1)
        
        # === DOMAIN FEATURES ===
        if domain:
            # Subdomain count
            parts = domain.split('.')
            features['subdomain_count'] = max(0, len(parts) - 2)
            
            # Domain has digits
            features['domain_has_digits'] = int(any(c.isdigit() for c in domain))
            
            # Domain has hyphens
            features['domain_has_hyphen'] = int('-' in domain)
            
            # TLD features
            tld = '.' + parts[-1] if len(parts) > 0 else ''
            features['has_suspicious_tld'] = int(tld in self.suspicious_tlds)
            features['has_trusted_tld'] = int(tld in self.trusted_tlds)
            features['tld_length'] = len(tld)
        else:
            features['subdomain_count'] = 0
            features['domain_has_digits'] = 0
            features['domain_has_hyphen'] = 0
            features['has_suspicious_tld'] = 0
            features['has_trusted_tld'] = 0
            features['tld_length'] = 0
        
        # === KEYWORD FEATURES ===
        features['has_phishing_keyword'] = int(any(kw in url for kw in self.phishing_keywords))
        features['num_phishing_keywords'] = sum(kw in url for kw in self.phishing_keywords)
        features['has_brand_name'] = int(any(brand in url for brand in self.targeted_brands))
        
        # === PATH FEATURES ===
        if path:
            features['path_has_extension'] = int(bool(re.search(r'\.(html|php|exe|zip)$', path)))
            features['suspicious_path'] = int(bool(re.search(r'(login|signin|verify|confirm|account)', path)))
        else:
            features['path_has_extension'] = 0
            features['suspicious_path'] = 0
        
        # === ENTROPY (measure of randomness) ===
        features['url_entropy'] = self._calculate_entropy(url)
        features['domain_entropy'] = self._calculate_entropy(domain) if domain else 0
        
        # === LENGTH CATEGORIES ===
        features['url_very_long'] = int(len(url) > 75)
        features['url_very_short'] = int(len(url) < 25)
        
        return list(features.values())
    
    def _calculate_entropy(self, text):
        """Calculate Shannon entropy of text"""
        if not text:
            return 0
        
        entropy = 0
        text_len = len(text)
        
        # Count character frequencies
        char_counts = {}
        for char in text:
            char_counts[char] = char_counts.get(char, 0) + 1
        
        # Calculate entropy
        for count in char_counts.values():
            probability = count / text_len
            entropy -= probability * np.log2(probability)
        
        return entropy
    
    def get_feature_names(self):
        """Return feature names for the extracted features"""
        return [
            'url_length', 'domain_length', 'path_length', 'query_length',
            'num_dots', 'num_hyphens', 'num_underscores', 'num_slashes',
            'num_questionmarks', 'num_equals', 'num_at', 'num_ampersands',
            'num_percent', 'num_digits', 'digit_ratio', 'special_char_ratio',
            'has_https', 'has_http', 'no_protocol', 'has_ip', 'has_at_symbol',
            'has_double_slash', 'subdomain_count', 'domain_has_digits',
            'domain_has_hyphen', 'has_suspicious_tld', 'has_trusted_tld',
            'tld_length', 'has_phishing_keyword', 'num_phishing_keywords',
            'has_brand_name', 'path_has_extension', 'suspicious_path',
            'url_entropy', 'domain_entropy', 'url_very_long', 'url_very_short'
        ]


def extract_features_from_csv(input_csv, output_csv):
    """
    Extract features from CSV containing URLs and save to new CSV.
    
    Args:
        input_csv: Path to CSV with 'url' and 'class' columns
        output_csv: Path to save extracted features
    """
    print(f"Loading dataset from: {input_csv}")
    df = pd.read_csv(input_csv)
    
    if 'url' not in df.columns:
        raise ValueError("CSV must contain 'url' column")
    
    print(f"Extracting features from {len(df)} URLs...")
    extractor = AdvancedURLFeatureExtractor()
    
    # Extract features
    features = extractor.transform(df['url'])
    feature_names = extractor.get_feature_names()
    
    # Create features DataFrame
    features_df = pd.DataFrame(features, columns=feature_names)
    
    # Add label column
    if 'class' in df.columns:
        features_df['class'] = df['class'].values
    
    # Save
    features_df.to_csv(output_csv, index=False)
    print(f"✅ Saved {len(features_df)} samples with {len(feature_names)} features to: {output_csv}")
    
    return features_df


if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parents[1]
    input_path = ROOT / "data" / "phishing_enhanced.csv"
    output_path = ROOT / "data" / "phishing_features.csv"
    
    if input_path.exists():
        extract_features_from_csv(input_path, output_path)
    else:
        print(f"❌ Input file not found: {input_path}")
        print("Run generate_enhanced_dataset.py first!")
