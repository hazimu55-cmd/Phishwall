import streamlit as st
import joblib
import pandas as pd
import os
from feature_extraction import extract_features

st.set_page_config(page_title="Phishing Detector", page_icon="🛡️", layout="centered")

st.markdown("""
<style>
    body, .stApp { background-color: #0f0f1a; color: #e0e0e0; }
    .result-box {
        border-radius: 10px;
        padding: 24px;
        text-align: center;
        font-size: 1.3rem;
        font-weight: bold;
        margin: 16px 0;
    }
    .safe    { background: #0d2b1d; border: 2px solid #00c853; color: #00e676; }
    .danger  { background: #2b0d0d; border: 2px solid #ff1744; color: #ff5252; }
</style>
""", unsafe_allow_html=True)


MODEL_PATH = "models/phishing_model.pkl"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)

model = load_model()


def rule_score(f):
    score = 0.0
    if f['phishing_keywords'] >= 2:  score += 0.25
    if f['phishing_keywords'] >= 4:  score += 0.25   # extra penalty for many keywords
    if f['num_at'] > 0:              score += 0.40   # @ in URL = almost always phishing
    if f['has_ip_address'] == 1:     score += 0.45   # raw IP = very suspicious
    if f['http_in_domain'] == 1:     score += 0.40   # http inside hostname = spoofing
    if f['is_https'] == 0:           score += 0.15   # no HTTPS = mild risk
    if f['num_subdomains'] > 2:      score += 0.20   # deep subdomains
    if f['url_length'] > 100:        score += 0.10   # very long URL
    if f['num_hyphens'] > 3:         score += 0.10   # many hyphens in domain
    return min(score, 1.0)


st.title("🛡️ Phishing URL Detector")
st.caption("Enter any URL to check if it's safe or a phishing attempt.")

if model is None:
    st.error("Model not found. Please run `train_model.py` first, then restart.")
    st.stop()


url_input = st.text_input("URL to check", placeholder="https://example.com/login")
analyze = st.button("Analyze", use_container_width=True)


if analyze:
    url = url_input.strip()
    if not url:
        st.warning("Please enter a URL.")
        st.stop()

    if not url.startswith("http"):
        url = "https://" + url

    features = extract_features(url)
    df = pd.DataFrame([features])

    ml_score  = model.predict_proba(df)[0][1]
    rules     = rule_score(features)
    final     = max(ml_score, rules)

    THRESHOLD  = 0.35
    is_phishing = final >= THRESHOLD

    st.divider()

    if is_phishing:
        st.markdown(f'<div class="result-box danger">🚨 Phishing Detected — {final*100:.0f}% risk</div>', unsafe_allow_html=True)
        st.error("Do not visit this site. It may steal your credentials or data.")
    else:
        st.markdown(f'<div class="result-box safe">✅ Looks Safe — {(1-final)*100:.0f}% confidence</div>', unsafe_allow_html=True)
        st.success("This URL appears legitimate. Still, stay cautious online.")

    st.markdown("**Phishing Risk**")
    st.progress(float(final))

    col1, col2, col3 = st.columns(3)
    col1.metric("ML Model Score",  f"{ml_score*100:.0f}%")
    col2.metric("Rule-based Score", f"{rules*100:.0f}%")
    col3.metric("Final Risk",       f"{final*100:.0f}%")

    with st.expander("What the detector looked at"):
        labels = {
            'url_length':        'URL length',
            'domain_length':     'Domain length',
            'path_length':       'Path length',
            'num_dots':          'Number of dots',
            'num_hyphens':       'Number of hyphens',
            'num_at':            'Has @ symbol',
            'num_digits':        'Digit count',
            'num_subdomains':    'Subdomain depth',
            'num_params':        'Query parameters',
            'num_obfuscated':    'Obfuscated chars (%xx)',
            'is_https':          'Uses HTTPS',
            'has_ip_address':    'Uses raw IP address',
            'http_in_domain':    '"http" inside hostname',
            'phishing_keywords': 'Phishing keywords found',
        }
        suspicious = {
            'num_at':            lambda v: v > 0,
            'has_ip_address':    lambda v: v == 1,
            'http_in_domain':    lambda v: v == 1,
            'phishing_keywords': lambda v: v >= 2,
            'url_length':        lambda v: v > 100,
            'num_subdomains':    lambda v: v > 2,
            'is_https':          lambda v: v == 0,
            'num_hyphens':       lambda v: v > 3,
        }
        rows = []
        for k, label in labels.items():
            val = features[k]
            flag = "⚠️" if k in suspicious and suspicious[k](val) else ""
            rows.append({"Feature": label, "Value": val, "": flag})
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


st.divider()
st.caption("ML model + rule-based hybrid detector")