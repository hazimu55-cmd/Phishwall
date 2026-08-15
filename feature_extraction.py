import re
import math
from collections import Counter
from urllib.parse import urlparse
import pandas as pd

ACTION_KEYWORDS = [
    'login', 'signin', 'sign-in', 'verify', 'secure', 'account',
    'update', 'confirm', 'banking', 'password', 'credential', 'wallet',
    'support', 'alert', 'suspended', 'limited', 'unusual', 'activity',
    'click', 'free', 'winner', 'prize', 'urgent',
    # India-specific scam/action terms (KYC-update, OTP, lottery, and
    # cashback/refund lures are extremely common in Indian SMS/WhatsApp
    # phishing — "smishing" — campaigns).
    'kyc', 'otp', 'lottery', 'cashback', 'reward', 'refund', 'recharge',
]

# These are meant to catch phishing sites IMPERSONATING these brands
# (e.g. "google-verify-security.tk"), not the brands' own real domains.
# They're checked separately in extract_features() so that a brand name
# appearing as the site's actual registered domain (google.com, paypal.com,
# ...) is never counted against it — only brand names showing up ANYWHERE
# ELSE (a different domain, a subdomain trick, the path) are suspicious.
BRAND_KEYWORDS = [
    'paypal', 'ebay', 'amazon', 'microsoft', 'apple',
    'google', 'facebook', 'netflix',
    # Indian banks, payment apps, and government services that are
    # heavily impersonated in Indian phishing/smishing campaigns.
    'sbi', 'icici', 'hdfc', 'axis', 'kotak', 'paytm', 'phonepe',
    'upi', 'aadhaar', 'aadhar', 'irctc', 'epfo', 'lic', 'jio', 'airtel',
    'vodafone',
]

# Kept for backwards compatibility with any code that still imports the old
# combined list name.
PHISHING_KEYWORDS = ACTION_KEYWORDS + BRAND_KEYWORDS

# ---------------------------------------------------------------------------
# TRUSTED DOMAINS — Indian market
# ---------------------------------------------------------------------------
# .gov.in / .nic.in are registration-restricted to verified Indian
# government bodies (NIC is the sole registrar). Since Nov 2025, the RBI
# has required Indian banks to migrate to the similarly-restricted
# ".bank.in" TLD specifically to curb bank-phishing — only RBI-verified
# banks can register it. A domain ending in one of these literally cannot
# be bought by a scammer, so it's treated as a hard trust signal rather
# than something the model has to learn statistically.
#
# This does NOT protect against a compromised legitimate site or a
# malicious link merely claiming to be from one of these domains in
# display text — it only verifies the URL's actual registered domain.
TRUSTED_TLD_SUFFIXES = ('.gov.in', '.nic.in', '.bank.in')

# Starter allowlist for well-known official Indian domains that do NOT
# fall under a restricted TLD above (e.g. mygov.in is a plain, open ".in"
# registration; rbi.org.in is ".org.in"). Expand this list as needed —
# it's intentionally small so every entry stays something you've verified
# yourself rather than something inherited unchecked.
TRUSTED_DOMAINS = {
    'mygov.in',
    'rbi.org.in',
}

# Two-part suffixes used to correctly find the REAL registrable label for
# Indian domains, e.g. 'sbi' in 'sbi.bank.in', not 'bank'. Without this,
# the brand-exemption logic below would treat "bank" as the root label of
# "sbi.bank.in" and incorrectly flag "sbi" as a suspicious brand mention
# on the bank's own legitimate site.
TWO_PART_SUFFIXES = {
    'gov.in', 'nic.in', 'bank.in', 'co.in', 'org.in',
    'net.in', 'ac.in', 'edu.in', 'res.in', 'gen.in', 'firm.in', 'ind.in',
}


def is_trusted_domain(domain: str) -> bool:
    """True if `domain` is a registration-restricted or explicitly
    allowlisted legitimate Indian domain (see TRUSTED_TLD_SUFFIXES /
    TRUSTED_DOMAINS above)."""
    d = domain.lower()
    if d in TRUSTED_DOMAINS or any(d.endswith('.' + td) for td in TRUSTED_DOMAINS):
        return True
    return any(d == suf.lstrip('.') or d.endswith(suf) for suf in TRUSTED_TLD_SUFFIXES)

# ---------------------------------------------------------------------------
# HIGH-RISK TLDS
# ---------------------------------------------------------------------------
# These extensions consistently show up in Spamhaus's Domain Reputation
# reports and CSC's TLD threat-frequency analyses as disproportionately
# abused for phishing/spam/malware — not because the extension itself is
# "bad", but because they're cheap, fast, and easy to register anonymously
# in bulk, which is exactly what a throwaway scam page needs. This is a
# probabilistic signal (most .xyz domains are NOT malicious), not a
# certainty like the trusted-TLD list above, so it nudges the score rather
# than overriding it outright. Abuse patterns shift over time — this list
# should be revisited periodically, not treated as permanent.
HIGH_RISK_TLDS = {
    'xyz', 'top', 'club', 'info', 'click', 'icu', 'cfd', 'sbs',
    'bond', 'work', 'loan', 'win', 'review', 'surf', 'quest', 'rest', 'cam',
    'tk', 'ml', 'ga', 'cf', 'gq',  # Freenom free ccTLDs, historically bulk-abused
}


def is_high_risk_tld(domain: str) -> bool:
    """True if `domain`'s TLD is one commonly associated with high phishing/
    spam abuse rates (see HIGH_RISK_TLDS above)."""
    labels = domain.lower().split('.')
    return bool(labels) and labels[-1] in HIGH_RISK_TLDS

IP_RE = re.compile(r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$')
DOMAIN_LABEL_RE = re.compile(r'^[a-z0-9]([a-z0-9-]{0,61}[a-z0-9])?$', re.IGNORECASE)
TLD_RE = re.compile(r'^[a-zA-Z]{2,24}$')


# ---------------------------------------------------------------------------
# ADVANCED FEATURE EXTRACTION FUNCTIONS
# ---------------------------------------------------------------------------

def calculate_shannon_entropy(text: str) -> float:
    """Calculate Shannon entropy of a string (randomness measure)."""
    if not text:
        return 0.0
    
    char_counts = Counter(text)
    total_chars = len(text)
    
    # Calculate probabilities and entropy
    probabilities = [count / total_chars for count in char_counts.values()]
    entropy = -sum(p * math.log2(p) for p in probabilities if p > 0)
    
    return entropy


def calculate_non_alphanumeric_entropy(text: str) -> float:
    """Calculate entropy of non-alphanumeric characters (critical for phishing detection)."""
    if not text:
        return 0.0
    
    # Extract only non-alphanumeric characters
    non_alnum_chars = [c for c in text if not c.isalnum()]
    
    if not non_alnum_chars:
        return 0.0
    
    char_counts = Counter(non_alnum_chars)
    total_chars = len(non_alnum_chars)
    
    probabilities = [count / total_chars for count in char_counts.values()]
    entropy = -sum(p * math.log2(p) for p in probabilities if p > 0)
    
    return entropy


def extract_char_ngrams(text: str, n: int) -> dict:
    """Extract character n-grams as frequency dictionary."""
    if len(text) < n:
        return {}
    
    ngrams = {}
    for i in range(len(text) - n + 1):
        ngram = text[i:i+n]
        ngrams[ngram] = ngrams.get(ngram, 0) + 1
    
    return ngrams


def get_character_category_ratios(text: str) -> dict:
    """Calculate ratios of different character categories."""
    if not text:
        return {'digit_ratio': 0, 'special_char_ratio': 0, 'letter_ratio': 0, 'upper_ratio': 0, 'lower_ratio': 0}
    
    total_chars = len(text)
    
    digits = sum(1 for c in text if c.isdigit())
    letters = sum(1 for c in text if c.isalpha())
    special = total_chars - digits - letters
    upper = sum(1 for c in text if c.isupper())
    lower = sum(1 for c in text if c.islower())
    
    return {
        'digit_ratio': digits / total_chars if total_chars > 0 else 0,
        'special_char_ratio': special / total_chars if total_chars > 0 else 0,
        'letter_ratio': letters / total_chars if total_chars > 0 else 0,
        'upper_ratio': upper / total_chars if total_chars > 0 else 0,
        'lower_ratio': lower / total_chars if total_chars > 0 else 0,
    }


def get_vowel_consonant_ratio(text: str) -> dict:
    """Calculate vowel to consonant ratio (for alphabetic characters)."""
    if not text:
        return {'vowel_ratio': 0, 'consonant_ratio': 0, 'vowel_consonant_ratio': 0}
    
    vowels = set('aeiouAEIOU')
    alphabetic_chars = [c for c in text if c.isalpha()]
    
    if not alphabetic_chars:
        return {'vowel_ratio': 0, 'consonant_ratio': 0, 'vowel_consonant_ratio': 0}
    
    vowel_count = sum(1 for c in alphabetic_chars if c in vowels)
    consonant_count = len(alphabetic_chars) - vowel_count
    
    return {
        'vowel_ratio': vowel_count / len(alphabetic_chars),
        'consonant_ratio': consonant_count / len(alphabetic_chars),
        'vowel_consonant_ratio': vowel_count / consonant_count if consonant_count > 0 else 0,
    }


def extract_url_token_tfidf_features(url: str) -> dict:
    """
    Extract simplified TF-IDF-like features for URL tokens.
    For a single URL, this returns term frequencies weighted by inverse
    document frequency assumptions based on common URL patterns.
    """
    # Tokenize URL by common delimiters
    tokens = re.split(r'[./\-_=?&]', url.lower())
    tokens = [t for t in tokens if t]  # Remove empty tokens
    
    if not tokens:
        return {'avg_token_tfidf': 0, 'max_token_tfidf': 0, 'unique_token_ratio': 0}
    
    # Calculate term frequencies
    token_counts = Counter(tokens)
    total_tokens = len(tokens)
    
    # Simplified IDF assumptions based on URL patterns
    # Common tokens in URLs have lower IDF, rare tokens have higher IDF
    common_url_tokens = {
        'http', 'https', 'www', 'com', 'org', 'net', 'html', 'php',
        'index', 'home', 'page', 'site', 'web', 'online', 'official'
    }
    
    tfidf_scores = []
    for token, count in token_counts.items():
        tf = count / total_tokens
        
        # Simplified IDF: lower for common tokens, higher for rare ones
        if token in common_url_tokens:
            idf = 1.0  # Low IDF for common tokens
        elif len(token) <= 2:
            idf = 1.5  # Medium IDF for short tokens
        else:
            idf = 2.0  # High IDF for longer, more specific tokens
        
        tfidf_scores.append(tf * idf)
    
    return {
        'avg_token_tfidf': sum(tfidf_scores) / len(tfidf_scores) if tfidf_scores else 0,
        'max_token_tfidf': max(tfidf_scores) if tfidf_scores else 0,
        'unique_token_ratio': len(token_counts) / total_tokens if total_tokens > 0 else 0,
    }


def extract_advanced_entropy_features(url: str) -> dict:
    """Extract all entropy-based features."""
    parsed = urlparse(url.lower())
    
    # Calculate entropy for different parts of URL
    full_entropy = calculate_shannon_entropy(url)
    domain_entropy = calculate_shannon_entropy(parsed.netloc)
    path_entropy = calculate_shannon_entropy(parsed.path)
    query_entropy = calculate_shannon_entropy(parsed.query)
    
    # Non-alphanumeric entropy (critical for phishing detection)
    full_nan_entropy = calculate_non_alphanumeric_entropy(url)
    domain_nan_entropy = calculate_non_alphanumeric_entropy(parsed.netloc)
    
    # Domain label entropy (for detecting random-looking domains)
    domain_labels = parsed.netloc.split('.')
    if domain_labels:
        max_label_entropy = max(calculate_shannon_entropy(label) for label in domain_labels)
        avg_label_entropy = sum(calculate_shannon_entropy(label) for label in domain_labels) / len(domain_labels)
    else:
        max_label_entropy = 0
        avg_label_entropy = 0
    
    return {
        'full_url_entropy': full_entropy,
        'domain_entropy': domain_entropy,
        'path_entropy': path_entropy,
        'query_entropy': query_entropy,
        'full_nan_entropy': full_nan_entropy,
        'domain_nan_entropy': domain_nan_entropy,
        'max_domain_label_entropy': max_label_entropy,
        'avg_domain_label_entropy': avg_label_entropy,
    }


def extract_ngram_features(url: str) -> dict:
    """Extract character n-gram features (2-4 grams)."""
    # Extract n-grams for different n values
    bigrams = extract_char_ngrams(url.lower(), 2)
    trigrams = extract_char_ngrams(url.lower(), 3)
    quadgrams = extract_char_ngrams(url.lower(), 4)
    
    # Calculate statistical properties of n-grams
    def ngram_stats(ngrams_dict):
        if not ngrams_dict:
            return {'count': 0, 'unique_ratio': 0, 'max_freq': 0}
        
        counts = list(ngrams_dict.values())
        total = sum(counts)
        unique = len(ngrams_dict)
        
        return {
            'count': total,
            'unique_ratio': unique / total if total > 0 else 0,
            'max_freq': max(counts) if counts else 0,
        }
    
    bigram_stats = ngram_stats(bigrams)
    trigram_stats = ngram_stats(trigrams)
    quadgram_stats = ngram_stats(quadgrams)
    
    return {
        'bigram_count': bigram_stats['count'],
        'bigram_unique_ratio': bigram_stats['unique_ratio'],
        'bigram_max_freq': bigram_stats['max_freq'],
        'trigram_count': trigram_stats['count'],
        'trigram_unique_ratio': trigram_stats['unique_ratio'],
        'trigram_max_freq': trigram_stats['max_freq'],
        'quadgram_count': quadgram_stats['count'],
        'quadgram_unique_ratio': quadgram_stats['unique_ratio'],
        'quadgram_max_freq': quadgram_stats['max_freq'],
    }


def is_valid_url(url: str) -> tuple[bool, str]:
    """
    Reject inputs that aren't actually shaped like a URL before they ever
    reach feature extraction / the model.

    Without this, a bare string like "ggkfjfjlk" gets treated as a domain.
    It has no dots, no TLD, no @ symbol, nothing — every red-flag feature
    comes back as 0, so it scores as confidently "safe" even though it
    isn't a URL at all. This checks structure first so garbage input gets
    flagged as invalid instead of silently scored as safe.
    """
    try:
        parsed = urlparse(url)
    except Exception:
        return False, "Could not parse this as a URL."

    if ' ' in url:
        return False, "URLs can't contain spaces."

    # netloc may include userinfo (user:pass@) and a port; strip both to
    # isolate the actual host for validation.
    domain = parsed.netloc.split('@')[-1].split(':')[0]

    if not domain:
        return False, "No domain found."

    if IP_RE.match(domain):
        return True, ""

    labels = domain.split('.')
    if len(labels) < 2:
        return False, "Missing a top-level domain (e.g. .com, .org)."

    tld = labels[-1]
    if not TLD_RE.match(tld):
        return False, "Invalid or missing top-level domain."

    for label in labels:
        if not label or not DOMAIN_LABEL_RE.match(label):
            return False, "Domain contains invalid characters."

    return True, ""


# Keep this list in sync with train_model.py's feature columns.
FEATURE_COLUMNS = [
    # Original features
    'url_length', 'domain_length', 'path_length',
    'num_dots', 'num_hyphens', 'num_at', 'num_digits',
    'num_subdomains', 'num_params', 'num_obfuscated',
    'is_https', 'has_ip_address', 'http_in_domain',
    'phishing_keywords', 'high_risk_tld',
    
    # New critical features (5 most impactful)
    'full_url_entropy',      # Strong randomness indicator
    'full_nan_entropy',      # Critical for obfuscation detection
    'domain_entropy',        # Random domain detection
    'digit_ratio',           # Character distribution analysis
    'special_char_ratio',    # Suspicious char concentration
]


def _root_domain_label(domain: str) -> str:
    """
    Best-effort registrable-domain label, e.g. 'google' for 'www.google.com',
    'accounts.google.com', or 'sbi' for 'sbi.bank.in'. Used only to decide
    whether a brand keyword is the site BEING that brand vs. a brand name
    showing up somewhere it shouldn't (a different domain, a subdomain
    trick, a lookalike).

    Two-part suffixes (.bank.in, .gov.in, .co.in, ...) need the label
    ONE level further up than single-part suffixes (.com, .in, .org) —
    otherwise 'sbi.bank.in' would incorrectly resolve to 'bank' instead
    of 'sbi'.
    """
    labels = domain.split('.')
    if len(labels) >= 3 and '.'.join(labels[-2:]) in TWO_PART_SUFFIXES:
        return labels[-3]
    if len(labels) >= 2:
        return labels[-2]
    return domain


def extract_features(url: str) -> dict:
    """Extract features for a single URL. Used for live single-URL prediction."""
    url = url.lower()
    parsed = urlparse(url)

    domain = parsed.netloc
    path = parsed.path
    full = url
    root_label = _root_domain_label(domain)

    action_hits = sum(1 for kw in ACTION_KEYWORDS if kw in full)
    # A brand keyword only counts as suspicious if it ISN'T the site's own
    # registered domain. "google" in "www.google.com" -> not counted.
    # "google" in "google-verify-security.tk" -> counted (root label is
    # "google-verify-security", not "google").
    brand_hits = sum(1 for kw in BRAND_KEYWORDS if kw in full and kw != root_label)

    # Extract original features
    features = {
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

        'phishing_keywords': action_hits + brand_hits,
        'high_risk_tld':     1 if is_high_risk_tld(domain) else 0,
    }

    # Add only the 5 critical new features
    features['full_url_entropy'] = calculate_shannon_entropy(full)
    features['full_nan_entropy'] = calculate_non_alphanumeric_entropy(full)
    features['domain_entropy'] = calculate_shannon_entropy(domain)
    
    # Character category ratios (only digit and special char ratios)
    char_ratios = get_character_category_ratios(full)
    features['digit_ratio'] = char_ratios['digit_ratio']
    features['special_char_ratio'] = char_ratios['special_char_ratio']

    return features