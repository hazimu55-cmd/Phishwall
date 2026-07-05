import re
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor
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
    'url_length', 'domain_length', 'path_length',
    'num_dots', 'num_hyphens', 'num_at', 'num_digits',
    'num_subdomains', 'num_params', 'num_obfuscated',
    'is_https', 'has_ip_address', 'http_in_domain',
    'phishing_keywords', 'high_risk_tld',
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

        'phishing_keywords': action_hits + brand_hits,
        'high_risk_tld':     1 if is_high_risk_tld(domain) else 0,
    }


def extract_features_batch(urls: list[str], max_workers: int = 8) -> pd.DataFrame:
    """
    Extract features for many URLs at once, returning a single DataFrame
    ready to hand to model.predict_proba() in ONE call.

    Why this fixes "not predicting in parallel":
    - The old app only ever built a 1-row DataFrame and called predict_proba
      once per URL, one click at a time. There was no batch path.
    - Feature extraction itself is cheap, pure-Python string/regex work, so it
      is CPU-bound and threads won't truly parallelize it (GIL). We still use
      a small ThreadPoolExecutor so this is a natural place to swap in real
      I/O-bound checks later (DNS lookups, WHOIS, blocklist API calls, etc.)
      without changing the calling code.
    - The actual speedup for prediction comes from calling
      model.predict_proba(df) ONCE on the whole batch, since scikit-learn
      vectorizes across rows internally (numpy under the hood). That is the
      "parallel" prediction you want, not a Python-level for-loop.
    """
    urls = [u.strip() for u in urls if u.strip()]
    if not urls:
        return pd.DataFrame(columns=FEATURE_COLUMNS)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        rows = list(executor.map(extract_features, urls))

    df = pd.DataFrame(rows, columns=FEATURE_COLUMNS)
    df.insert(0, 'url', urls)
    return df