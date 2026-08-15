import re
from urllib.parse import urlparse
import streamlit as st
import streamlit.components.v1 as components
import joblib
import pandas as pd
import os
from feature_extraction import extract_features, is_valid_url, is_trusted_domain, FEATURE_COLUMNS

st.set_page_config(page_title="PhishWall", page_icon="🛡️", layout="centered")

# ---------------------------------------------------------------------------
# ICONS — small inline SVGs, stroke=currentColor so CSS controls color/glow.
# Purely presentational; nothing here touches prediction logic.
# ---------------------------------------------------------------------------
ICON_SHIELD_LOCK = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6"><path d="M12 2.5l8 3v6c0 5-3.4 8.6-8 10-4.6-1.4-8-5-8-10v-6l8-3z"/><rect x="9" y="11" width="6" height="5" rx="1"/><path d="M10.3 11V9.3a1.7 1.7 0 0 1 3.4 0V11"/></svg>'
ICON_GLOBE = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6"><circle cx="12" cy="12" r="9"/><path d="M3 12h18M12 3a14 14 0 0 1 0 18 14 14 0 0 1 0-18z"/></svg>'
ICON_LINK = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7"><path d="M9 15l6-6M10 6l1-1a4 4 0 0 1 6 6l-1 1M14 18l-1 1a4 4 0 0 1-6-6l1-1"/></svg>'
ICON_SHIELD_CHECK = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6"><path d="M12 2.5l8 3v6c0 5-3.4 8.6-8 10-4.6-1.4-8-5-8-10v-6l8-3z"/><path d="M9 12.2l2 2 4-4.4"/></svg>'
ICON_SEARCH = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="10.5" cy="10.5" r="6.5"/><path d="M20 20l-4.8-4.8"/></svg>'
ICON_WARNING = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6"><path d="M12 3l10 18H2L12 3z"/><path d="M12 10v4.5"/><circle cx="12" cy="17.3" r="0.9" fill="currentColor" stroke="none"/></svg>'
ICON_BOLT = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6"><path d="M13 2 4 14h6l-1 8 9-12h-6l1-8z"/></svg>'
ICON_LOCK = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6"><rect x="5" y="11" width="14" height="9" rx="2"/><path d="M8 11V7a4 4 0 0 1 8 0v4"/></svg>'
ICON_FINGERPRINT = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.4"><path d="M12 2.5l8 3v6c0 5-3.4 8.6-8 10-4.6-1.4-8-5-8-10v-6l8-3z"/><path d="M12 8a3.2 3.2 0 0 0-3.2 3.2c0 3-1 4.6-2.3 5.9M12 8a3.2 3.2 0 0 1 3.2 3.2c0 1-.1 1.8-.4 2.6M9.6 17.4c1-1.1 1.6-2.4 1.6-4.2a1.6 1.6 0 1 1 3.2 0c0 .5 0 1-.1 1.4"/></svg>'
ICON_BADGE_CHECK = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6"><path d="M12 2l2.4 2 3-.6.6 3 2.6 1.6-1.6 2.6.6 3-3 .6L14 17l-2 2-2-2-3 .6-.6-3-2.6-1.6 1.6-2.6-.6-3 3-.6L10 4z"/><path d="M9 12.2l2 2 4-4.4"/></svg>'

# ---------------------------------------------------------------------------
# THEME — glassmorphism + neon-cyber SOC dashboard, matching provided ref.
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@600;700;800;900&family=Rajdhani:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root {
    --bg:          #04070a;
    --panel:       rgba(7, 20, 13, 0.55);
    --panel-solid: #081410;
    --border:      rgba(70, 255, 140, 0.28);
    --border-soft: rgba(70, 255, 140, 0.14);
    --neon:        #4dff86;
    --neon-strong: #9dffc2;
    --neon-dim:    #1c8a4d;
    --cyan:        #5be8ff;
    --danger:      #ff4d5e;
    --danger-dim:  #7a1c26;
    --text:        #eafff2;
    --text-dim:    #8fb8a2;
}

html, body, .stApp {
    background: radial-gradient(ellipse 80% 60% at 50% 0%, #0b1c12 0%, #04070a 45%, #020402 100%) !important;
    color: var(--text);
    font-family: 'Rajdhani', sans-serif;
}

/* hide default streamlit chrome so it reads as a standalone product */
#MainMenu, footer, header, [data-testid="stToolbar"],
[data-testid="stDecoration"], [data-testid="stStatusWidget"] { visibility: hidden; height: 0; }
.block-container { padding-top: 2.2rem !important; padding-bottom: 3rem !important; max-width: 880px !important; }

/* vignette + faint scanlines for atmosphere — kept behind content via negative z-index */
.stApp::before {
    content: ""; position: fixed; inset: 0; pointer-events: none; z-index: -1;
    background:
        repeating-linear-gradient(to bottom, rgba(77,255,134,0.02) 0px, rgba(77,255,134,0.02) 1px, transparent 1px, transparent 3px),
        radial-gradient(ellipse 100% 60% at 50% 0%, transparent 40%, rgba(0,0,0,0.55) 100%);
}
/* Negative z-index is spec-guaranteed to paint behind normal in-flow content,
   regardless of what Streamlit's internal wrapper class names happen to be
   in the installed version — unlike z-index:0, which can end up ABOVE
   non-positioned content and hide the entire app behind the canvas. */
#pw-bg-canvas { position: fixed !important; inset: 0; z-index: -1; }
[data-testid="stAppViewContainer"], [data-testid="stMain"], .stApp {
    position: relative; z-index: 1; background: transparent !important;
}

h1, h2, h3 { font-family: 'Orbitron', sans-serif !important; }

/* ---------- header ---------- */
.pw-header { text-align: center; margin-bottom: 6px; }
.pw-header-icon { color: var(--neon); width: 46px; height: 46px; margin: 0 auto 6px;
    filter: drop-shadow(0 0 10px rgba(77,255,134,0.85)); }
.pw-title-frame { position: relative; display: inline-block; padding: 14px 42px; margin-top: 4px; }
.pw-title-frame::before, .pw-title-frame::after,
.pw-title-frame .pw-corner-tr, .pw-title-frame .pw-corner-br {
    content: ""; position: absolute; width: 22px; height: 22px; border-color: rgba(200,255,225,0.55);
}
.pw-title-frame::before { top: 0; left: 0; border-top: 2px solid; border-left: 2px solid; }
.pw-title-frame::after  { bottom: 0; left: 0; border-bottom: 2px solid; border-left: 2px solid; }
.pw-title-frame .pw-corner-tr { top: 0; right: 0; border-top: 2px solid rgba(200,255,225,0.55); border-right: 2px solid rgba(200,255,225,0.55); }
.pw-title-frame .pw-corner-br { bottom: 0; right: 0; border-bottom: 2px solid rgba(200,255,225,0.55); border-right: 2px solid rgba(200,255,225,0.55); }

.pw-title {
    font-family: 'Orbitron', sans-serif; font-weight: 900;
    font-size: clamp(2.4rem, 7vw, 4.2rem); letter-spacing: 6px;
    color: var(--neon-strong); margin: 0; line-height: 1;
    text-shadow: 0 0 8px rgba(157,255,194,0.9), 0 0 26px rgba(77,255,134,0.75), 0 0 60px rgba(77,255,134,0.4);
    animation: pulseGlow 3.2s ease-in-out infinite;
}
@keyframes pulseGlow {
    0%, 100% { text-shadow: 0 0 8px rgba(157,255,194,0.9), 0 0 26px rgba(77,255,134,0.75), 0 0 60px rgba(77,255,134,0.4); }
    50%      { text-shadow: 0 0 12px rgba(157,255,194,1), 0 0 38px rgba(77,255,134,0.9), 0 0 80px rgba(77,255,134,0.55); }
}
.pw-sub {
    font-family: 'Rajdhani', sans-serif; font-weight: 600; letter-spacing: 5px;
    text-transform: uppercase; color: var(--text-dim); font-size: 0.85rem; margin-top: 14px;
}
.pw-divider { width: 140px; height: 2px; margin: 22px auto 28px; border-radius: 2px;
    background: linear-gradient(90deg, transparent, var(--neon), transparent); opacity: 0.8; }

/* ---------- glass cards ---------- */
.pw-card {
    background: var(--panel); border: 1px solid var(--border); border-radius: 20px;
    padding: 26px 28px; margin-bottom: 22px; backdrop-filter: blur(14px);
    box-shadow: 0 0 30px rgba(0,0,0,0.35), inset 0 0 40px rgba(77,255,134,0.03);
    animation: fadeSlideUp 0.5s ease both;
}
@keyframes fadeSlideUp { from { opacity: 0; transform: translateY(14px); } to { opacity: 1; transform: translateY(0); } }

.pw-card-head { display: flex; align-items: center; gap: 12px; margin-bottom: 18px; }
.pw-icon-box { width: 36px; height: 36px; min-width: 36px; border: 1px solid var(--border); border-radius: 10px;
    display: flex; align-items: center; justify-content: center; color: var(--neon); background: rgba(77,255,134,0.05); }
.pw-icon-box svg { width: 18px; height: 18px; }
.pw-card-head span {
    font-family: 'Rajdhani', sans-serif; font-weight: 700; letter-spacing: 3px; text-transform: uppercase;
    color: var(--neon-strong); font-size: 1.05rem;
}

.pw-helper { display: flex; align-items: center; gap: 8px; margin: 12px 2px 20px; color: var(--text-dim); font-size: 0.92rem; }
.pw-helper svg { width: 16px; height: 16px; color: var(--neon-dim); min-width: 16px; }

/* url input with inline link icon */
.pw-field-wrap { position: relative; }
.pw-field-wrap svg { position: absolute; left: 16px; top: 50%; transform: translateY(-50%);
    width: 18px; height: 18px; color: var(--neon); z-index: 3; pointer-events: none; }
.pw-field-wrap div[data-testid="stTextInput"] input { padding-left: 46px !important; }

div[data-testid="stTextInput"] input, div[data-testid="stTextArea"] textarea {
    background: rgba(3, 10, 6, 0.75) !important; color: var(--neon-strong) !important;
    border: 1px solid var(--border) !important; border-radius: 14px !important;
    font-family: 'JetBrains Mono', monospace !important; font-size: 1.02rem !important;
    caret-color: var(--neon); padding-top: 14px !important; padding-bottom: 14px !important;
}
div[data-testid="stTextInput"] input:focus, div[data-testid="stTextArea"] textarea:focus {
    border-color: var(--neon) !important; box-shadow: 0 0 0 3px rgba(77,255,134,0.15), 0 0 20px rgba(77,255,134,0.25) !important;
}
div[data-testid="stTextInput"] input::placeholder, div[data-testid="stTextArea"] textarea::placeholder { color: #4a6b58 !important; }

div[data-testid="stFileUploaderDropzone"] {
    background: rgba(3, 10, 6, 0.5) !important; border: 1px dashed var(--border) !important; border-radius: 14px !important;
}

/* buttons — big neon pill */
.stButton button, .stDownloadButton button {
    background: linear-gradient(90deg, var(--neon-dim), var(--neon) 55%, var(--neon-strong)) !important;
    color: #04160c !important; font-family: 'Rajdhani', sans-serif !important; font-weight: 700 !important;
    letter-spacing: 3px !important; text-transform: uppercase !important; font-size: 1.02rem !important;
    border: none !important; border-radius: 999px !important; padding: 0.7rem 0 !important;
    box-shadow: 0 0 22px rgba(77,255,134,0.35); transition: transform 0.15s ease, box-shadow 0.15s ease;
}
.stButton button:hover, .stDownloadButton button:hover {
    transform: translateY(-2px) scale(1.01); box-shadow: 0 0 34px rgba(77,255,134,0.55);
}
.stButton button:active, .stDownloadButton button:active { transform: translateY(0) scale(0.99); }

/* ---------- awaiting / result state card ---------- */
.pw-state-card {
    display: flex; align-items: center; gap: 22px; padding: 30px 30px;
}
.pw-state-text h3 {
    font-family: 'Orbitron', sans-serif; font-weight: 700; letter-spacing: 2px; font-size: 1.25rem; margin: 0 0 8px;
}
.pw-state-text p { color: var(--text-dim); margin: 0; font-size: 0.95rem; line-height: 1.5; }

.pw-radar { width: 70px; height: 70px; min-width: 70px; border-radius: 50%; position: relative;
    background: conic-gradient(from 0deg, rgba(77,255,134,0.85), transparent 65%);
    animation: spin 2.6s linear infinite; }
.pw-radar::before { content: ""; position: absolute; inset: 6px; border-radius: 50%; background: var(--bg);
    border: 1px solid rgba(77,255,134,0.25); }
.pw-radar::after { content: ""; position: absolute; inset: 30px; border-radius: 50%; background: var(--neon);
    box-shadow: 0 0 8px var(--neon); }
@keyframes spin { to { transform: rotate(360deg); } }

.pw-state-deco { margin-left: auto; opacity: 0.16; color: var(--neon); }
.pw-state-deco svg { width: 60px; height: 60px; }

.pw-result-icon { width: 56px; height: 56px; min-width: 56px; }
.pw-result-icon.safe svg { color: var(--neon); filter: drop-shadow(0 0 14px rgba(77,255,134,0.8)); }
.pw-result-icon.danger svg { color: var(--danger); filter: drop-shadow(0 0 14px rgba(255,77,94,0.8)); animation: warnPulse 1.1s ease-in-out infinite; }
@keyframes warnPulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.55; } }

.pw-state-card.safe   { border-color: rgba(77,255,134,0.5); box-shadow: 0 0 34px rgba(77,255,134,0.16), inset 0 0 40px rgba(77,255,134,0.05); }
.pw-state-card.danger { border-color: rgba(255,77,94,0.55); box-shadow: 0 0 34px rgba(255,77,94,0.18), inset 0 0 40px rgba(255,77,94,0.06); }
.pw-state-card.safe .pw-state-text h3   { color: var(--neon-strong); }
.pw-state-card.danger .pw-state-text h3 { color: #ffb3ba; }

[data-testid="stMetric"] {
    background: var(--panel); border: 1px solid var(--border-soft); border-radius: 14px; padding: 12px;
    backdrop-filter: blur(10px);
}
[data-testid="stMetricLabel"] { color: var(--text-dim) !important; font-family: 'Rajdhani', sans-serif !important; letter-spacing: 1px; }
[data-testid="stMetricValue"] { color: var(--neon-strong) !important; font-family: 'JetBrains Mono', monospace !important; }

.stProgress > div > div { background: linear-gradient(90deg, var(--neon-dim), var(--neon)) !important; }

.stTabs [data-baseweb="tab-list"] { gap: 4px; background: var(--panel); padding: 4px; border-radius: 14px;
    border: 1px solid var(--border-soft); }
.stTabs [data-baseweb="tab"] { font-family: 'Rajdhani', sans-serif; font-weight: 600; letter-spacing: 1px;
    color: var(--text-dim); border-radius: 10px !important; }
.stTabs [aria-selected="true"] { color: var(--neon-strong) !important; background: rgba(77,255,134,0.1) !important; }

/* feature badge row */
.pw-badges { display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; margin-top: 8px; }
.pw-badge {
    background: var(--panel); border: 1px solid var(--border-soft); border-radius: 16px; padding: 16px 14px;
    display: flex; gap: 10px; align-items: flex-start; backdrop-filter: blur(10px);
    transition: transform 0.15s ease, border-color 0.15s ease;
}
.pw-badge:hover { transform: translateY(-3px); border-color: var(--border); }
.pw-badge .pw-icon-box { width: 30px; height: 30px; min-width: 30px; }
.pw-badge .pw-icon-box svg { width: 15px; height: 15px; }
.pw-badge-title { font-family: 'Rajdhani', sans-serif; font-weight: 700; letter-spacing: 1px; font-size: 0.8rem;
    color: var(--neon-strong); text-transform: uppercase; }
.pw-badge-sub { font-family: 'Rajdhani', sans-serif; font-size: 0.78rem; color: var(--text-dim); margin-top: 2px; }

.pw-footer { text-align: center; color: var(--text-dim); font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem; letter-spacing: 1px; margin-top: 30px; opacity: 0.7; }

@media (max-width: 640px) {
    .pw-badges { grid-template-columns: repeat(2, 1fr); }
    .pw-state-card { flex-direction: column; text-align: center; }
    .pw-state-deco { margin-left: 0; }
    .pw-title { letter-spacing: 3px; }
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# BACKGROUND — thousands-feel field of drifting binary particles on canvas,
# injected into the parent document so it sits full-viewport behind every
# Streamlit element. Purely decorative; degrades gracefully if blocked.
# ---------------------------------------------------------------------------
components.html("""
<script>
(function() {
    try {
        const doc = window.parent.document;
        let canvas = doc.getElementById('pw-bg-canvas');
        if (!canvas) {
            canvas = doc.createElement('canvas');
            canvas.id = 'pw-bg-canvas';
            doc.body.prepend(canvas);
        }
        const ctx = canvas.getContext('2d');
        function resize() {
            canvas.width = window.parent.innerWidth;
            canvas.height = window.parent.innerHeight;
        }
        resize();
        window.parent.addEventListener('resize', resize);

        const COUNT = 220;
        const particles = [];
        for (let i = 0; i < COUNT; i++) {
            particles.push({
                x: Math.random() * canvas.width,
                y: Math.random() * canvas.height,
                vx: (Math.random() - 0.5) * 0.35,
                vy: (Math.random() - 0.5) * 0.35,
                size: 9 + Math.random() * 15,
                alpha: 0.12 + Math.random() * 0.45,
                char: Math.random() < 0.5 ? '0' : '1',
                flipAt: 600 + Math.random() * 2400,
                t: Math.random() * 2400,
            });
        }

        function tick() {
            ctx.fillStyle = 'rgba(4, 7, 10, 0.4)';
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            for (const p of particles) {
                p.x += p.vx; p.y += p.vy; p.t += 1;
                if (p.x < -10) p.x = canvas.width + 10;
                if (p.x > canvas.width + 10) p.x = -10;
                if (p.y < -10) p.y = canvas.height + 10;
                if (p.y > canvas.height + 10) p.y = -10;
                if (p.t > p.flipAt) { p.char = p.char === '0' ? '1' : '0'; p.t = 0; }
                ctx.globalAlpha = p.alpha;
                ctx.fillStyle = '#4dff86';
                ctx.font = p.size + 'px "JetBrains Mono", monospace';
                ctx.fillText(p.char, p.x, p.y);
            }
            ctx.globalAlpha = 1;
            window.parent.requestAnimationFrame(tick);
        }
        tick();
    } catch (e) { /* fail silently — CSS gradient background still shows */ }
})();
</script>
""", height=0)


MODEL_PATH = "models/phishing_model.pkl"
THRESHOLD = 0.35


@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)


def normalize_url(raw: str) -> str:
    """
    Default missing scheme to http:// (NOT https://).
    Forcing https on every bare URL silently maxes out the `is_https`
    feature for inputs that never claimed to be secure, which quietly
    destroyed that signal for the model. Defaulting to http is the more
    conservative / suspicious assumption when the scheme is unknown.
    """
    raw = raw.strip()
    if not raw:
        return raw
    if not re.match(r'^[a-zA-Z]+://', raw):
        return "http://" + raw
    return raw


def rule_score(f: dict) -> float:
    score = 0.0
    # Original features
    if f['phishing_keywords'] >= 1:  score += 0.15
    if f['phishing_keywords'] >= 2:  score += 0.15
    if f['phishing_keywords'] >= 4:  score += 0.20
    if f['high_risk_tld'] == 1:      score += 0.30  # e.g. .xyz, .top, .click — see HIGH_RISK_TLDS
    if f['num_at'] > 0:              score += 0.40
    if f['has_ip_address'] == 1:     score += 0.45
    if f['http_in_domain'] == 1:     score += 0.40
    if f['is_https'] == 0:           score += 0.15
    if f['num_subdomains'] > 2:      score += 0.20
    if f['url_length'] > 100:        score += 0.10
    if f['num_hyphens'] > 3:         score += 0.10
    
    # New critical features (5 most impactful)
    if f['full_url_entropy'] > 4.5:       score += 0.15  # High randomness
    if f['full_nan_entropy'] > 2.0:       score += 0.20  # High special char randomness (critical)
    if f['domain_entropy'] > 4.0:         score += 0.10  # Random domain
    if f['digit_ratio'] > 0.15:           score += 0.10  # High digit concentration
    if f['special_char_ratio'] > 0.25:    score += 0.15  # High special char concentration
    
    return min(score, 1.0)


def score_dataframe(df: pd.DataFrame, model) -> pd.DataFrame:
    """One vectorized predict_proba call for the whole batch — this is the
    'parallel' scoring path, replacing per-row calls in a loop."""
    feat_only = df[FEATURE_COLUMNS]
    ml_scores = pd.Series(model.predict_proba(feat_only)[:, 1], index=df.index)
    rule_scores = feat_only.apply(rule_score, axis=1)
    final = pd.concat([ml_scores, rule_scores], axis=1).max(axis=1)

    out = df.copy()
    out['ml_score'] = ml_scores
    out['rule_score'] = rule_scores
    out['final_risk'] = final
    out['is_phishing'] = out['final_risk'] >= THRESHOLD

    # Registration-restricted Indian government/bank domains (or explicit
    # allowlist entries) are verified directly rather than left to the
    # model's statistical judgment — see is_trusted_domain().
    trusted = out['url'].apply(lambda u: is_trusted_domain(urlparse(u).netloc))
    out['trusted_domain'] = trusted
    out.loc[trusted, 'final_risk'] = 0.0
    out.loc[trusted, 'is_phishing'] = False
    return out


model = load_model()

# ---------------------------------------------------------------------------
# HEADER
# ---------------------------------------------------------------------------
st.markdown(f"""
<div class="pw-header">
    <div class="pw-header-icon">{ICON_SHIELD_LOCK}</div>
    <div class="pw-title-frame">
        <span class="pw-corner-tr"></span><span class="pw-corner-br"></span>
        <h1 class="pw-title">PHISHWALL</h1>
    </div>
    <div class="pw-sub">AI-Powered Phishing URL Detection</div>
</div>
<div class="pw-divider"></div>
""", unsafe_allow_html=True)

if model is None:
    st.error("Model not found. Run `train_model.py` first, then restart.")
    st.stop()

# ---------------------------------------------------------------------------
# SINGLE URL ANALYSIS (batch processing removed)
# ---------------------------------------------------------------------------
st.markdown(f'''
<div class="pw-card">
    <div class="pw-card-head">
        <div class="pw-icon-box">{ICON_GLOBE}</div>
        <span>Enter URL to Analyze</span>
    </div>
    <div class="pw-field-wrap">
''', unsafe_allow_html=True)

url_input = st.text_input("URL to check", placeholder="https://example.com",
                           label_visibility="collapsed")

st.markdown(f'''
    </div>
    <div class="pw-helper">{ICON_SHIELD_CHECK}<span>We analyze the URL and predict if it's safe or a phishing attempt.</span></div>
''', unsafe_allow_html=True)

analyze = st.button("🔍  Analyze URL  →", use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)  # close pw-card

if analyze:
    url = normalize_url(url_input)
    if not url:
        st.warning("Please enter a URL.")
        st.stop()

    valid, reason = is_valid_url(url)
    if not valid:
        st.warning(f"⚠️ That doesn't look like a valid URL. {reason}")
        st.stop()

    domain = urlparse(url).netloc
    if is_trusted_domain(domain):
        st.markdown(f'''
        <div class="pw-card pw-state-card safe">
            <div class="pw-result-icon safe">{ICON_BADGE_CHECK}</div>
            <div class="pw-state-text">
                <h3>✅ VERIFIED OFFICIAL DOMAIN</h3>
                <p>This domain is registration-restricted to verified Indian government/banking
                entities (or is on our verified allowlist), so we can confirm it's legitimate
                directly — no model guessing needed. This only verifies the domain itself, not
                whether the site's content or links are safe.</p>
            </div>
        </div>
        ''', unsafe_allow_html=True)
        st.stop()

    with st.spinner("scanning..."):
        features = extract_features(url)
        df = pd.DataFrame([features])[FEATURE_COLUMNS]
        ml_score = model.predict_proba(df)[0][1]
        rules = rule_score(features)
        final = max(ml_score, rules)
        is_phishing = final >= THRESHOLD

    if is_phishing:
        st.markdown(f'''
        <div class="pw-card pw-state-card danger">
            <div class="pw-result-icon danger">{ICON_WARNING}</div>
            <div class="pw-state-text">
                <h3>🚨 PHISHING DETECTED — {final*100:.0f}% RISK</h3>
                <p>Do not visit this site. It may steal your credentials or data.</p>
            </div>
        </div>
        ''', unsafe_allow_html=True)
    else:
        st.markdown(f'''
        <div class="pw-card pw-state-card safe">
            <div class="pw-result-icon safe">{ICON_SHIELD_CHECK}</div>
            <div class="pw-state-text">
                <h3>✅ LOOKS SAFE — {(1-final)*100:.0f}% CONFIDENCE</h3>
                <p>This URL appears legitimate. Still, stay cautious online.</p>
            </div>
        </div>
        ''', unsafe_allow_html=True)

        st.markdown("**Phishing Risk**")
        st.progress(float(final))

        col1, col2, col3 = st.columns(3)
        col1.metric("ML Model Score", f"{ml_score*100:.0f}%")
        col2.metric("Rule-based Score", f"{rules*100:.0f}%")
        col3.metric("Final Risk", f"{final*100:.0f}%")

        with st.expander("What the detector looked at"):
            labels = {
                # Original features
                'url_length': 'URL length', 'domain_length': 'Domain length', 'path_length': 'Path length',
                'num_dots': 'Number of dots', 'num_hyphens': 'Number of hyphens', 'num_at': 'Has @ symbol',
                'num_digits': 'Digit count', 'num_subdomains': 'Subdomain depth', 'num_params': 'Query parameters',
                'num_obfuscated': 'Obfuscated chars (%xx)', 'is_https': 'Uses HTTPS',
                'has_ip_address': 'Uses raw IP address', 'http_in_domain': '"http" inside hostname',
                'phishing_keywords': 'Phishing keywords found',
                # New critical features
                'full_url_entropy': 'Full URL entropy', 'full_nan_entropy': 'Non-alphanumeric entropy',
                'domain_entropy': 'Domain entropy', 'digit_ratio': 'Digit ratio', 'special_char_ratio': 'Special char ratio',
            }
            suspicious = {
                'num_at': lambda v: v > 0, 'has_ip_address': lambda v: v == 1,
                'http_in_domain': lambda v: v == 1, 'phishing_keywords': lambda v: v >= 2,
                'url_length': lambda v: v > 100, 'num_subdomains': lambda v: v > 2,
                'is_https': lambda v: v == 0, 'num_hyphens': lambda v: v > 3,
                'full_url_entropy': lambda v: v > 4.5, 'full_nan_entropy': lambda v: v > 2.0,
                'domain_entropy': lambda v: v > 4.0, 'digit_ratio': lambda v: v > 0.15, 'special_char_ratio': lambda v: v > 0.25,
            }
            rows = [{"Feature": label, "Value": f"{features[k]:.4f}" if isinstance(features[k], float) else features[k],
                     "": "⚠️" if k in suspicious and suspicious[k](features[k]) else ""}
                    for k, label in labels.items()]
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

else:
    st.markdown(f'''
    <div class="pw-card pw-state-card">
        <div class="pw-radar"></div>
        <div class="pw-state-text">
            <h3 style="color:var(--neon-strong)">AWAITING ANALYSIS</h3>
            <p>Enter a URL above and click Analyze to see if it's safe or a phishing threat.</p>
        </div>
        <div class="pw-state-deco">{ICON_FINGERPRINT}</div>
    </div>
    ''', unsafe_allow_html=True)

    st.markdown(f'''
    <div class="pw-badges">
        <div class="pw-badge"><div class="pw-icon-box">{ICON_SHIELD_CHECK}</div>
            <div><div class="pw-badge-title">ML Powered</div><div class="pw-badge-sub">Advanced Machine Learning</div></div></div>
        <div class="pw-badge"><div class="pw-icon-box">{ICON_BOLT}</div>
            <div><div class="pw-badge-title">Real-Time</div><div class="pw-badge-sub">Instant URL Analysis</div></div></div>
        <div class="pw-badge"><div class="pw-icon-box">{ICON_SHIELD_CHECK}</div>
            <div><div class="pw-badge-title">Accurate</div><div class="pw-badge-sub">High Detection Accuracy</div></div></div>
        <div class="pw-badge"><div class="pw-icon-box">{ICON_LOCK}</div>
            <div><div class="pw-badge-title">Secure</div><div class="pw-badge-sub">Your Safety, Our Priority</div></div></div>
    </div>
    ''', unsafe_allow_html=True)

