import re
from urllib.parse import urlparse

PHISHING_KEYWORDS = [
    'login', 'signin', 'sign-in', 'verify', 'secure', 'account',
    'update', 'confirm', 'banking', 'paypal', 'ebay', 'amazon',
    'microsoft', 'apple', 'google', 'facebook', 'netflix', 'password',
    'credential', 'wallet', 'support', 'alert', 'suspended', 'limited',
    'unusual', 'activity', 'click', 'free', 'winner', 'prize', 'urgent'
]

def extract_features(url):
    url = url.lower()
    parsed = urlparse(url)

    domain = parsed.netloc
    path = parsed.path
    full = url

    return {
        'url_length':       len(full),
        'domain_length':    len(domain),
        'path_length':      len(path),

        'num_dots':         full.count('.'),
        'num_hyphens':      full.count('-'),
        'num_at':           full.count('@'),         # @ in URL = always suspicious
        'num_digits':       sum(c.isdigit() for c in full),
        'num_subdomains':   max(len(domain.split('.')) - 2, 0),
        'num_params':       len(parsed.query.split('&')) if parsed.query else 0,
        'num_obfuscated':   full.count('%'),          # %xx encoding = suspicious

        'is_https':         1 if parsed.scheme == 'https' else 0,
        'has_ip_address':   1 if re.match(r'^\d+\.\d+\.\d+\.\d+', domain) else 0,
        'http_in_domain':   1 if 'http' in domain else 0,   # http inside hostname = spoofing

        'phishing_keywords': sum(1 for kw in PHISHING_KEYWORDS if kw in full),
    }