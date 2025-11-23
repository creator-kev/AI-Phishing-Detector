# generate_sample_dataset.py
"""
Creates a realistic sample phishing dataset for testing.
Save this to data/phishing.csv
"""

import pandas as pd

# Legitimate URLs (Class = 0)
legitimate_urls = [
    "https://www.google.com",
    "https://www.facebook.com",
    "https://www.amazon.com",
    "https://github.com",
    "https://stackoverflow.com",
    "https://www.wikipedia.org",
    "https://www.reddit.com",
    "https://www.youtube.com",
    "https://www.linkedin.com",
    "https://www.twitter.com",
    "https://www.microsoft.com",
    "https://www.apple.com",
    "https://www.netflix.com",
    "https://www.bbc.com",
    "https://www.cnn.com",
    "https://docs.google.com/document",
    "https://drive.google.com/file",
    "https://mail.google.com",
    "https://www.paypal.com",
    "https://www.chase.com",
    "https://www.wellsfargo.com",
    "https://www.bankofamerica.com",
    "https://www.instagram.com",
    "https://www.tiktok.com",
    "https://www.adobe.com",
    "https://aws.amazon.com",
    "https://azure.microsoft.com",
    "https://cloud.google.com",
    "https://www.shopify.com",
    "https://www.ebay.com",
    "https://www.walmart.com",
    "https://www.target.com",
    "https://www.bestbuy.com",
    "https://www.nike.com",
    "https://www.adidas.com",
    "https://www.espn.com",
    "https://www.nytimes.com",
    "https://www.wsj.com",
    "https://www.forbes.com",
    "https://www.medium.com",
    "https://mmu-course-hub.vercel.app",
]

# Phishing URLs (Class = 1)
phishing_urls = [
    "http://paypal-secure.tk/login",
    "http://amazon-verify.xyz/account",
    "http://192.168.1.1/admin",
    "https://bank-security-update.com/signin",
    "http://microsoft-account-verify.tk",
    "http://secure-paypal-login.tk",
    "http://apple-id-verify.com/login",
    "http://bank-alert.xyz/verify",
    "http://account-verify-amazon.com",
    "http://192.168.0.1/login",
    "http://netflix-billing-update.tk/payment",
    "http://facebook-security-check.xyz/verify",
    "http://google-account-recovery.tk/signin",
    "http://chase-bank-alert.com/secure",
    "http://wellsfargo-verify-account.xyz/login",
    "http://187.234.12.45/admin/login",
    "http://secure-login-paypal.tk/verify",
    "http://amazon-security-alert.com/update",
    "http://microsoft-support-verify.xyz/account",
    "http://apple-security-check.tk/signin",
    "http://account-suspended-paypal.com/restore",
    "http://urgent-netflix-billing.xyz/update",
    "http://verify-your-facebook.tk/security",
    "http://google-unusual-activity.com/verify",
    "http://bank-of-america-alert.xyz/signin",
    "http://203.45.67.89/secure/login",
    "http://instagram-verify-account.tk/security",
    "http://twitter-account-suspended.xyz/appeal",
    "http://linkedin-security-check.tk/verify",
    "http://dropbox-storage-full.com/upgrade",
    "http://icloud-storage-upgrade.xyz/payment",
    "http://spotify-premium-expired.tk/renew",
    "http://steam-account-verification.com/secure",
    "http://ebay-seller-verification.xyz/confirm",
    "http://walmart-gift-card-winner.tk/claim",
    "http://target-account-verify.com/signin",
    "http://bestbuy-order-confirmation.xyz/track",
    "http://fedex-package-delivery.tk/track",
    "http://ups-delivery-failed.com/reschedule",
    "http://dhl-customs-payment.xyz/pay",
"https://secure-login.bank-example.test/account/signin?user=alice&redirect=https%3A%2F%2Fexample.com",
    "http://login.verify-example.test@safe.example.com/update",
    "http://192.0.2.123/confirm.php?ref=ABCD-1234&email=support@example.test",
    "https://accounts.example.test.secure-login.example.com/wp-admin/login.php",
    "http://very-long-subdomain.lots.of.dots.subdomain.example.test/path/to/resource/with/a/very/long/name",

]

# Create DataFrame
data = []

for url in legitimate_urls:
    data.append({"url": url, "class": 0})

for url in phishing_urls:
    data.append({"url": url, "class": 1})

df = pd.DataFrame(data)

# Shuffle
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Save
output_path = "data/phishing.csv"
df.to_csv(output_path, index=False)

print(f"✅ Created dataset: {output_path}")
print(f"📊 Total samples: {len(df)}")
print(f"📋 Legitimate: {(df['class'] == 0).sum()}")
print(f"⚠️  Phishing: {(df['class'] == 1).sum()}")
print(f"\nFirst 5 rows:")
print(df.head())
