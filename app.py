# ============================================================
# EXPLAINABLE STUDENT OUTCOME PREDICTOR
# XGBOOST + SHAP + STREAMLIT
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Student Outcome Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# DESIGN SYSTEM — "AURORA GLASS"
# Violet-to-pink glass panels on a soft gradient backdrop,
# echoing the floating dual-card mockup: rounded glass tiles,
# a warm coral primary action (the "play button" accent), and
# soft glowing orbs instead of a hard grid.
#
# IMPORTANT RENDERING NOTE:
# Every HTML block below is written with NO blank lines inside
# it and tags flush left. A blank line inside an
# st.markdown(unsafe_allow_html=True) string breaks it into a
# new block, and Markdown then treats indented continuations as
# a code block instead of HTML — that was the original bug.
#
# ANIMATION NOTE:
# Streamlit reruns the whole script on every widget change, so
# an animation on every element would replay on every keystroke.
# Entrance animations here are deliberately scoped: the
# letterhead/index cards animate once per session (gated by
# st.session_state), and the verdict/ledger/case-notes animate
# each time "Generate Full Case Report" is pressed, since that's
# the moment a replay actually reads as feedback, not flicker.
# ============================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Space+Grotesk:wght@400;500;600;700&display=swap');
:root{--text:#fdf5ff;--muted:rgba(255,255,255,.70);--violet:#8b7bf0;--pink:#e26fc4;--coral:#ff5d7a;--gold:#ffcf6f;--mint:#3fe8b0;--glass:rgba(255,255,255,.10);--glass-strong:rgba(255,255,255,.16);--line:rgba(255,255,255,.24);--shadow:0 20px 60px rgba(40,14,70,.45)}
@keyframes glassIn{from{opacity:0;transform:translateY(10px)}to{opacity:1;transform:translateY(0)}}
@keyframes floatGlow{0%,100%{transform:translateY(0);opacity:.45}50%{transform:translateY(-10px);opacity:.7}}
@keyframes pulse{0%{box-shadow:0 0 0 0 rgba(63,232,176,.45)}70%{box-shadow:0 0 0 9px rgba(63,232,176,0)}100%{box-shadow:0 0 0 0 rgba(63,232,176,0)}}
@keyframes needleSweep{from{transform:rotate(-90deg)}to{transform:rotate(var(--target))}}
html,body,[class*="css"]{font-family:'Inter',sans-serif}
.stApp{color:var(--text);background:radial-gradient(circle at 10% 6%,rgba(139,123,240,.55),transparent 42%),radial-gradient(circle at 88% 10%,rgba(226,111,196,.48),transparent 45%),radial-gradient(circle at 55% 95%,rgba(255,150,190,.32),transparent 52%),linear-gradient(160deg,#221749 0%,#432a75 40%,#7a3f86 72%,#a94f8c 100%);min-height:100vh}
.stApp::before{content:"";position:fixed;inset:0;pointer-events:none;background:radial-gradient(ellipse at 50% -10%,rgba(255,255,255,.14),transparent 55%);z-index:0}
.stApp::after{content:"";position:fixed;width:560px;height:560px;right:-200px;bottom:-200px;border-radius:50%;background:radial-gradient(circle,rgba(255,168,208,.35),transparent 68%);filter:blur(12px);pointer-events:none;animation:floatGlow 8s ease-in-out infinite;z-index:0}
.block-container{padding-top:1.15rem;padding-bottom:3rem;max-width:1420px;position:relative;z-index:1}
section[data-testid="stSidebar"]{background:linear-gradient(180deg,rgba(34,21,68,.90),rgba(64,29,78,.86));border-right:1px solid rgba(255,255,255,.14);box-shadow:18px 0 55px rgba(20,8,40,.30);backdrop-filter:blur(26px)}
section[data-testid="stSidebar"] *{color:var(--text)!important}
section[data-testid="stSidebar"] h2,section[data-testid="stSidebar"] h3{font-family:'Space Grotesk',sans-serif!important}
section[data-testid="stSidebar"] hr{border-color:rgba(255,255,255,.16)!important}
section[data-testid="stSidebar"] div[data-testid="stMetricValue"]{color:var(--coral)!important;font-family:'Space Grotesk',sans-serif!important}
section[data-testid="stSidebar"] div[data-testid="stAlert"]{background:rgba(255,255,255,.10)!important;border:1px solid rgba(255,255,255,.22)!important;border-radius:16px}
.letterhead{position:relative;overflow:visible;padding:24px 26px 22px;margin-bottom:16px;border:1px solid var(--line);border-radius:26px;background:linear-gradient(150deg,rgba(255,255,255,.16),rgba(255,255,255,.05));box-shadow:var(--shadow),inset 0 1px 0 rgba(255,255,255,.20);backdrop-filter:blur(26px);animation:glassIn .55s ease both}
.letterhead::before{content:"";position:absolute;inset:12px -16px -16px 12px;border-radius:26px;background:linear-gradient(150deg,rgba(255,255,255,.07),rgba(255,255,255,.02));border:1px solid rgba(255,255,255,.12);z-index:-1;transform:rotate(-2deg)}
.letterhead::after{content:"";position:absolute;width:280px;height:280px;right:-110px;top:-160px;border-radius:50%;background:radial-gradient(circle,rgba(255,168,208,.32),transparent 68%);filter:blur(10px)}
.letterhead-crest{display:inline-flex;align-items:center;justify-content:center;width:52px;height:52px;border-radius:18px;margin-right:14px;vertical-align:middle;background:linear-gradient(145deg,rgba(255,255,255,.32),rgba(255,255,255,.08));border:1px solid rgba(255,255,255,.38);box-shadow:inset 0 1px 0 rgba(255,255,255,.28),0 10px 26px rgba(40,10,60,.35);font-size:23px}
.letterhead-eyebrow{font-size:10px;letter-spacing:2.3px;color:#ffc3e6;text-transform:uppercase;margin-bottom:9px;font-weight:700}
.letterhead-title{font-family:'Space Grotesk',sans-serif;font-weight:800;font-size:38px;color:#fffbff;display:inline-block;vertical-align:middle}
.letterhead-rule{height:1px;background:linear-gradient(90deg,rgba(255,93,122,.75),rgba(226,111,196,.55),transparent);margin:14px 0 11px}
.letterhead-sub{color:var(--muted);font-size:14px;max-width:820px}
.index-row{display:flex;gap:14px;margin:20px 0 27px;flex-wrap:wrap}
.index-card{flex:1;min-width:190px;position:relative;overflow:hidden;padding:17px 18px 14px;border:1px solid var(--line);border-radius:20px;background:linear-gradient(150deg,rgba(255,255,255,.13),rgba(255,255,255,.04));box-shadow:var(--shadow),inset 0 1px 0 rgba(255,255,255,.16);backdrop-filter:blur(22px);transition:transform .18s ease,border-color .18s ease}
.index-card::after{content:"";position:absolute;width:120px;height:120px;right:-55px;top:-55px;border-radius:50%;background:radial-gradient(circle,rgba(139,123,240,.30),transparent 68%)}
.index-card:nth-child(2)::after{background:radial-gradient(circle,rgba(226,111,196,.28),transparent 68%)}
.index-card:nth-child(3)::after{background:radial-gradient(circle,rgba(255,207,111,.26),transparent 68%)}
.index-card:nth-child(4)::after{background:radial-gradient(circle,rgba(63,232,176,.24),transparent 68%)}
.index-card:hover{transform:translateY(-3px);border-color:rgba(255,255,255,.42)}
.index-label{font-size:10px;letter-spacing:1.8px;color:rgba(255,255,255,.72);text-transform:uppercase;font-weight:700}
.index-value{font-family:'Space Grotesk',sans-serif;font-weight:700;font-size:30px;color:#fffbff;margin-top:5px}
.index-caption{font-size:11.5px;color:rgba(255,255,255,.55);margin-top:3px}
.tab-title{font-family:'Space Grotesk',sans-serif;font-weight:700;font-size:21px;color:#fffbff;margin-top:7px;margin-bottom:5px}
.tab-chip{font-size:9px;letter-spacing:1.4px;background:rgba(255,255,255,.14);color:#ffd3ec;border:1px solid rgba(255,255,255,.26);padding:4px 8px;border-radius:8px;margin-right:9px;vertical-align:2px}
.section-description{color:rgba(255,255,255,.60);font-size:12.5px;margin-bottom:14px}
.stTabs [data-baseweb="tab-list"]{gap:7px;border-bottom:1px solid rgba(255,255,255,.16);padding-bottom:7px}
.stTabs [data-baseweb="tab"]{font-family:'Space Grotesk',sans-serif;font-size:12px;font-weight:600;color:rgba(255,255,255,.68);background:rgba(255,255,255,.07);border:1px solid rgba(255,255,255,.14);border-radius:12px;padding:8px 13px;backdrop-filter:blur(16px)}
.stTabs [data-baseweb="tab"]:hover{color:#fffbff;border-color:rgba(255,255,255,.32);background:rgba(255,255,255,.12)}
.stTabs [aria-selected="true"]{color:#fffbff!important;background:linear-gradient(150deg,rgba(255,255,255,.22),rgba(255,255,255,.08))!important;border-color:rgba(255,150,190,.55)!important;box-shadow:inset 0 1px 0 rgba(255,255,255,.25),0 8px 22px rgba(255,93,122,.20)!important;backdrop-filter:blur(18px)}
div[data-baseweb="input"],div[data-baseweb="select"],div[data-baseweb="base-input"]{background:transparent!important}
div[data-baseweb="input"]>div,div[data-baseweb="select"]>div,div[data-baseweb="base-input"]{min-height:38px!important;height:38px!important;background:linear-gradient(145deg,rgba(255,255,255,.16),rgba(255,255,255,.06))!important;border:1px solid rgba(255,255,255,.26)!important;border-radius:12px!important;box-shadow:inset 0 1px 0 rgba(255,255,255,.12),0 5px 18px rgba(20,6,36,.16)!important;backdrop-filter:blur(18px)}
div[data-baseweb="input"]>div:hover,div[data-baseweb="select"]>div:hover{border-color:rgba(255,150,190,.55)!important}
div[data-baseweb="input"]:focus-within>div,div[data-baseweb="select"]:focus-within>div{border-color:rgba(255,93,122,.60)!important;box-shadow:0 0 0 1px rgba(255,93,122,.28),0 7px 22px rgba(20,6,36,.22)!important}
div[data-baseweb="input"] input,div[data-baseweb="select"] input,div[data-baseweb="base-input"] input{background:transparent!important;height:36px!important;font-family:'Space Grotesk',sans-serif!important;font-size:13px!important;color:#fffbff!important}
div[data-baseweb="select"] span{color:#fffbff!important;font-size:13px!important}
label[data-testid="stWidgetLabel"] p{font-size:11.5px!important;color:rgba(255,255,255,.68)!important;font-weight:600;margin-bottom:3px!important}
div[data-testid="stNumberInput"]{background:transparent!important}
div[data-testid="stNumberInput"]>div{background:linear-gradient(145deg,rgba(255,255,255,.16),rgba(255,255,255,.06))!important;border:1px solid rgba(255,255,255,.26)!important;border-radius:12px!important;overflow:hidden;backdrop-filter:blur(18px)}
div[data-testid="stNumberInput"] input{background:transparent!important;border:none!important}
div[data-testid="stNumberInput"] button{color:#ffd3ec!important;background:rgba(255,255,255,.10)!important;border:0!important;border-left:1px solid rgba(255,255,255,.16)!important}
div[data-testid="stNumberInput"] button:hover{background:rgba(255,150,190,.24)!important;color:#fff!important}
div[data-baseweb="popover"]{background:rgba(36,20,58,.97)!important;border:1px solid rgba(255,255,255,.24)!important;border-radius:14px!important;box-shadow:0 24px 70px rgba(20,6,36,.55)!important;backdrop-filter:blur(22px)}
div[data-baseweb="menu"]{background:rgba(36,20,58,.97)!important}
div[role="option"]{color:#fdf0ff!important;background:transparent!important;font-size:12px!important}
div[role="option"]:hover{background:rgba(255,150,190,.18)!important}
div[role="option"][aria-selected="true"]{background:rgba(255,93,122,.22)!important}
div.stButton>button{width:100%;min-height:46px;border-radius:13px;font-family:'Space Grotesk',sans-serif;font-size:11px;font-weight:700;letter-spacing:1.2px;text-transform:uppercase;border:1px solid rgba(255,255,255,.38);background:linear-gradient(135deg,var(--coral),var(--pink));color:#fffbff;box-shadow:inset 0 1px 0 rgba(255,255,255,.30),0 14px 34px rgba(255,93,122,.35);backdrop-filter:blur(18px)}
div.stButton>button:hover{transform:translateY(-2px);border-color:rgba(255,255,255,.55);box-shadow:inset 0 1px 0 rgba(255,255,255,.35),0 18px 40px rgba(255,93,122,.48)}
.live-strip{display:flex;align-items:center;gap:13px;background:linear-gradient(150deg,rgba(255,255,255,.13),rgba(255,255,255,.04));border:1px solid rgba(255,255,255,.22);border-radius:16px;padding:11px 16px;margin:17px 0 21px;box-shadow:inset 0 1px 0 rgba(255,255,255,.10),0 12px 35px rgba(20,6,36,.20);backdrop-filter:blur(20px)}
.live-dot{width:8px;height:8px;border-radius:50%;background:var(--mint);animation:pulse 1.8s infinite;flex-shrink:0}
.live-label{font-size:9px;letter-spacing:1.8px;color:#ffd3ec;font-weight:700;flex-shrink:0}
.live-track{flex:1;height:7px;background:rgba(255,255,255,.14);border-radius:6px;overflow:hidden;min-width:70px}
.live-fill{height:100%;border-radius:6px}.live-fill.ink-crimson{background:linear-gradient(90deg,#ff5d7a,#ff3b5c)}.live-fill.ink-brass{background:linear-gradient(90deg,#ffcf6f,#ffe29a)}.live-fill.ink-forest{background:linear-gradient(90deg,#3fe8b0,#69efb3)}
.live-value{font-size:11.5px;color:rgba(255,255,255,.78);white-space:nowrap}
.verdict{display:flex;align-items:center;gap:30px;background:linear-gradient(150deg,rgba(255,255,255,.14),rgba(255,255,255,.04));border:1px solid rgba(255,255,255,.24);border-radius:22px;padding:25px 30px;margin:17px 0 22px;flex-wrap:wrap;box-shadow:var(--shadow),inset 0 1px 0 rgba(255,255,255,.18);backdrop-filter:blur(26px)}
.gauge-wrap{text-align:center;flex-shrink:0}.gauge{position:relative;width:210px;height:106px;overflow:hidden}.gauge-arc{position:absolute;inset:0;border-radius:210px 210px 0 0;background:conic-gradient(from 270deg at 50% 100%,#3fe8b0 0deg,#8b7bf0 42deg,#ffcf6f 90deg,#ff5d7a 180deg)}
.gauge-mask{position:absolute;left:24px;right:24px;bottom:0;height:82px;background:#2c1c50;border-radius:180px 180px 0 0}.gauge-needle{position:absolute;left:50%;bottom:0;width:3px;height:92px;background:#fffbff;transform-origin:bottom center;border-radius:3px;box-shadow:0 0 9px rgba(255,251,255,.85);animation:needleSweep 1.1s cubic-bezier(.34,1.56,.64,1) both}.gauge-center{position:absolute;left:50%;bottom:-7px;width:16px;height:16px;margin-left:-8px;background:#ff5d7a;border-radius:50%;border:2px solid #2c1c50;box-shadow:0 0 13px rgba(255,93,122,.55)}
.gauge-readout{font-size:12px;color:rgba(255,255,255,.70);margin-top:9px}.gauge-readout b{color:#fffbff;font-size:14px}.verdict-stamp{flex:1;min-width:220px}.stamp-eyebrow{font-size:9px;letter-spacing:2.2px;color:#ffb8dd;text-transform:uppercase;font-weight:700}
.stamp-outcome{font-family:'Space Grotesk',sans-serif;font-weight:800;font-size:32px;margin:8px 0 10px;display:inline-block;padding:6px 16px;border:1px solid;border-radius:13px;background:rgba(255,255,255,.08)}.stamp-outcome.ink-crimson{color:var(--coral);border-color:rgba(255,93,122,.55)}.stamp-outcome.ink-brass{color:var(--gold);border-color:rgba(255,207,111,.52)}.stamp-outcome.ink-forest{color:var(--mint);border-color:rgba(63,232,176,.52)}
.stamp-risk{font-size:10px;font-weight:700;letter-spacing:1.8px}.stamp-risk.ink-crimson{color:var(--coral)}.stamp-risk.ink-brass{color:var(--gold)}.stamp-risk.ink-forest{color:var(--mint)}
.ledger,.explorer-box{background:linear-gradient(150deg,rgba(255,255,255,.12),rgba(255,255,255,.04));border:1px solid rgba(255,255,255,.20);border-radius:18px;padding:18px 22px;margin-bottom:6px;box-shadow:inset 0 1px 0 rgba(255,255,255,.10),0 12px 35px rgba(20,6,36,.18);backdrop-filter:blur(22px)}
.ledger-row{display:flex;align-items:center;gap:13px;padding:8px 0;border-bottom:1px solid rgba(255,255,255,.12)}.ledger-row:last-child{border-bottom:none}.ledger-label{font-size:10px;letter-spacing:1.3px;color:rgba(255,255,255,.72);width:95px;flex-shrink:0;font-weight:700}.ledger-track{flex:1;height:8px;background:rgba(255,255,255,.14);border-radius:6px;overflow:hidden}.ledger-fill{height:100%;border-radius:6px}.ledger-fill.ink-crimson{background:linear-gradient(90deg,#ff5d7a,#ff3b5c)}.ledger-fill.ink-brass{background:linear-gradient(90deg,#f6c76c,#ffe29a)}.ledger-fill.ink-forest{background:linear-gradient(90deg,#3fe8b0,#69efb3)}.ledger-value{font-family:'Space Grotesk',sans-serif;font-weight:700;font-size:11px;width:52px;text-align:right;color:#fffbff}
.compare-card{background:linear-gradient(150deg,rgba(255,255,255,.13),rgba(255,255,255,.04));border:1px solid rgba(255,255,255,.22);border-radius:18px;padding:18px 20px;text-align:center;box-shadow:inset 0 1px 0 rgba(255,255,255,.10),0 12px 30px rgba(20,6,36,.20);backdrop-filter:blur(22px)}
.compare-name{font-size:9px;letter-spacing:1.8px;color:#ffc3e6;text-transform:uppercase;margin-bottom:7px;font-weight:700}.compare-outcome{font-family:'Space Grotesk',sans-serif;font-weight:800;font-size:24px;margin-bottom:4px}.compare-outcome.ink-crimson{color:var(--coral)}.compare-outcome.ink-brass{color:var(--gold)}.compare-outcome.ink-forest{color:var(--mint)}.compare-sub{font-size:11px;color:rgba(255,255,255,.68)}.compare-delta{text-align:center;font-size:11.5px;color:rgba(255,255,255,.80);margin-top:12px;padding:10px;background:rgba(255,255,255,.08);border:1px solid rgba(255,255,255,.16);border-radius:12px}
div[data-testid="stDataFrame"]{border:1px solid rgba(255,255,255,.22)!important;border-radius:15px;overflow:hidden;box-shadow:0 12px 32px rgba(20,6,36,.20)}
.case-notes{background:linear-gradient(150deg,rgba(255,255,255,.12),rgba(255,255,255,.04));border:1px solid rgba(255,255,255,.20);border-left:3px solid var(--coral);border-radius:14px;padding:15px 19px;margin-top:12px}.case-note-row{font-size:12.5px;padding:5px 0;color:rgba(255,255,255,.78)}.case-note-row b{color:#fffbff}.footer{text-align:center;color:rgba(255,255,255,.50);font-size:10px;padding-top:28px;padding-bottom:7px;border-top:1px solid rgba(255,255,255,.14);margin-top:28px;letter-spacing:.6px}
header[data-testid="stHeader"]{background:rgba(30,17,58,.55)!important;backdrop-filter:blur(16px)}[data-testid="stDecoration"]{display:none}[data-testid="stToolbar"]{opacity:.45}
</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD DATASET
# ============================================================

@st.cache_data
def load_dataset():
    data = pd.read_excel("data.csv.xlsx")
    return data


# ============================================================
# TRAIN MODEL
#
# IMPORTANT:
# No model is passed as an argument to a cached function.
# This avoids the Streamlit UnhashableParamError.
# ============================================================

@st.cache_resource
def train_model():

    df = load_dataset()

    X = df.drop("Target", axis=1)
    y_text = df["Target"]

    encoder = LabelEncoder()
    y = encoder.fit_transform(y_text)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        random_state=42,
        eval_metric="mlogloss"
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    return (
        model, encoder, X, X_train, X_test,
        y_train, y_test, accuracy
    )


# ============================================================
# LOAD EVERYTHING
# ============================================================

try:
    df = load_dataset()

    (
        model, label_encoder, X, X_train, X_test,
        y_train, y_test, accuracy
    ) = train_model()

except Exception as e:
    st.error("Unable to load the application.")
    st.exception(e)
    st.stop()


# ============================================================
# SHARED PREDICTION HELPER
# Used by the live strip, the full report, the What-If explorer,
# and the Student A/B comparison — all call the same model, just
# on different input dictionaries. No model logic is duplicated.
# ============================================================

def run_prediction(values_dict, defaults_dict):
    row = {}
    for feature in X.columns:
        row[feature] = values_dict.get(feature, defaults_dict[feature])
    input_df = pd.DataFrame([row], columns=X.columns)

    pred = model.predict(input_df)[0]
    probs = model.predict_proba(input_df)[0]
    label = label_encoder.inverse_transform([pred])[0]

    prob_dict = {}
    for i, cls in enumerate(label_encoder.classes_):
        prob_dict[cls] = probs[i] * 100

    dropout = prob_dict.get("Dropout", 0)
    return pred, label, prob_dict, dropout


def risk_tier(dropout_probability):
    if dropout_probability >= 60:
        return "HIGH RISK", "ink-crimson"
    elif dropout_probability >= 30:
        return "MODERATE RISK", "ink-brass"
    else:
        return "LOW RISK", "ink-forest"


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown("## 🎓 Student Outcome AI")
    st.markdown(
        "An explainable machine-learning system "
        "for predicting student academic outcomes."
    )
    st.divider()
    st.markdown("### 📊 Model Information")
    st.write(f"**Dataset:** {len(df):,} students")
    st.write(f"**Features:** {X.shape[1]}")
    st.write(f"**Training samples:** {len(X_train):,}")
    st.write(f"**Testing samples:** {len(X_test):,}")
    st.write(f"**Model:** XGBoost")
    st.write(f"**Explainability:** SHAP")
    st.divider()
    st.markdown("### 🎯 Model Performance")
    st.metric("Accuracy", f"{accuracy * 100:.2f}%")
    st.divider()
    st.info(
        "Adjust the student's information in any tab — the "
        "**Live Read** strip updates instantly. Click "
        "**Generate Full Case Report** for the complete SHAP "
        "breakdown."
    )


# ============================================================
# SESSION STATE — entrance-animation gate + compare slots
# ============================================================

if "loaded_once" not in st.session_state:
    st.session_state.loaded_once = True

if "profile_a" not in st.session_state:
    st.session_state.profile_a = None

if "profile_b" not in st.session_state:
    st.session_state.profile_b = None


# ============================================================
# HEADER — LETTERHEAD
# ============================================================

st.markdown(f"""
<div class="letterhead"><div class="letterhead-eyebrow">OFFICE OF INSTITUTIONAL ANALYTICS · CASE FILE MODEL</div><span class="letterhead-crest">🎓</span><span class="letterhead-title">Student Outcome Predictor</span><div class="letterhead-rule"></div><div class="letterhead-sub">AI-assisted review of a student's academic record, with every prediction accompanied by a transparent, factor-by-factor SHAP explanation.</div></div>
""", unsafe_allow_html=True)


# ============================================================
# OVERVIEW — INDEX CARDS
# ============================================================

st.markdown(f"""
<div class="index-row"><div class="index-card"><div class="index-label">Students</div><div class="index-value">{len(df):,}</div><div class="index-caption">Dataset records</div></div><div class="index-card"><div class="index-label">Features</div><div class="index-value">{X.shape[1]}</div><div class="index-caption">Student attributes</div></div><div class="index-card"><div class="index-label">Model</div><div class="index-value">XGB</div><div class="index-caption">Multiclass classifier</div></div><div class="index-card"><div class="index-label">Accuracy</div><div class="index-value">{accuracy * 100:.2f}%</div><div class="index-caption">Test-set performance</div></div></div>
""", unsafe_allow_html=True)


# ============================================================
# STUDENT INPUT
# ============================================================

st.markdown('<div class="tab-title"><span class="tab-chip">FILE 01</span>Student Profile</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-description">Provide the student attributes used by the machine-learning model. The Live Read below updates as you go.</div>',
    unsafe_allow_html=True
)


# ============================================================
# DEFAULT VALUES
# ============================================================

defaults = {}

for column in X.columns:
    if pd.api.types.is_numeric_dtype(X[column]):
        if X[column].dtype.kind in "iu":
            defaults[column] = int(X[column].mode()[0])
        else:
            defaults[column] = float(X[column].median())


# ============================================================
# FEATURE GROUPS
# ============================================================

personal_features = [
    "Marital status",
    "Gender",
    "Age at enrollment",
    "International",
    "Displaced",
    "Educational special needs"
]

academic_features = [
    "Course",
    "Application mode",
    "Application order",
    "Previous qualification",
    "Previous qualification (grade)",
    "Admission grade",
    "Curricular units 1st sem (credited)",
    "Curricular units 1st sem (enrolled)",
    "Curricular units 1st sem (evaluations)",
    "Curricular units 1st sem (approved)",
    "Curricular units 1st sem (grade)",
    "Curricular units 1st sem (without evaluations)",
    "Curricular units 2nd sem (credited)",
    "Curricular units 2nd sem (enrolled)",
    "Curricular units 2nd sem (evaluations)",
    "Curricular units 2nd sem (approved)",
    "Curricular units 2nd sem (grade)",
    "Curricular units 2nd sem (without evaluations)"
]

financial_features = [
    "Debtor",
    "Tuition fees up to date",
    "Scholarship holder"
]

family_features = [
    "Mother's qualification",
    "Father's qualification",
    "Mother's occupation",
    "Father's occupation"
]

economic_features = [
    "Unemployment rate",
    "Inflation rate",
    "GDP"
]

attendance_features = [
    "Daytime/evening attendance\t"
]


# ============================================================
# INPUT FUNCTION
# ============================================================

student_values = {}


def create_inputs(feature_list):

    available = [f for f in feature_list if f in X.columns]

    columns = st.columns(2)

    for index, feature in enumerate(available):
        with columns[index % 2]:

            value = defaults[feature]

            if X[feature].dtype.kind in "iu":
                min_value = int(X[feature].min())
                max_value = int(X[feature].max())

                student_values[feature] = st.number_input(
                    feature.replace("\t", ""),
                    min_value=min_value,
                    max_value=max_value,
                    value=int(value),
                    step=1,
                    key="input_" + feature
                )

            else:
                min_value = float(X[feature].min())
                max_value = float(X[feature].max())

                student_values[feature] = st.number_input(
                    feature.replace("\t", ""),
                    min_value=min_value,
                    max_value=max_value,
                    value=float(value),
                    key="input_" + feature
                )


# ============================================================
# TABBED INPUT LAYOUT
# ============================================================

tab_labels = [
    "👤 Personal",
    "📚 Academic",
    "💳 Financial",
    "🏠 Family",
    "🌍 Economic",
    "🕐 Attendance"
]

profile_tabs = st.tabs(tab_labels)

with profile_tabs[0]:
    create_inputs(personal_features)

with profile_tabs[1]:
    create_inputs(academic_features)

with profile_tabs[2]:
    create_inputs(financial_features)

with profile_tabs[3]:
    create_inputs(family_features)

with profile_tabs[4]:
    create_inputs(economic_features)

with profile_tabs[5]:
    create_inputs(attendance_features)


# ============================================================
# LIVE READ STRIP — updates on every input change, no button
# ============================================================

_, live_label, live_probs, live_dropout = run_prediction(student_values, defaults)
live_risk_text, live_ink = risk_tier(live_dropout)

st.markdown(f"""
<div class="live-strip"><div class="live-dot"></div><div class="live-label">LIVE READ</div><div class="live-track"><div class="live-fill {live_ink}" style="width:{live_dropout}%"></div></div><div class="live-value">{live_dropout:.1f}% dropout likelihood → currently reads as <b>{live_label}</b> ({live_risk_text})</div></div>
""", unsafe_allow_html=True)


# ============================================================
# PREDICTION BUTTON
# ============================================================

predict_button = st.button(
    "◆  Generate Full Case Report",
    use_container_width=True
)


# ============================================================
# FULL CASE REPORT
# ============================================================

if predict_button:

    input_data = {}
    for feature in X.columns:
        input_data[feature] = student_values.get(feature, defaults[feature])
    input_df = pd.DataFrame([input_data], columns=X.columns)

    prediction, predicted_label, probability_dict, dropout_probability = run_prediction(student_values, defaults)

    # ========================================================
    # RESULT — GAUGE + VERDICT STAMP
    # ========================================================

    st.markdown('<div class="tab-title"><span class="tab-chip">FILE 02</span>Prediction Result</div>', unsafe_allow_html=True)

    risk_text, ink_class = risk_tier(dropout_probability)
    needle_angle = -90 + (dropout_probability / 100) * 180

    st.markdown(f"""
<div class="verdict"><div class="gauge-wrap"><div class="gauge"><div class="gauge-arc"></div><div class="gauge-mask"></div><div class="gauge-needle" style="--target:{needle_angle}deg"></div><div class="gauge-center"></div></div><div class="gauge-readout">Dropout likelihood&nbsp;<b>{dropout_probability:.1f}%</b></div></div><div class="verdict-stamp"><div class="stamp-eyebrow">Predicted academic outcome</div><div class="stamp-outcome {ink_class}">{predicted_label}</div><div class="stamp-risk {ink_class}">{risk_text}</div></div></div>
""", unsafe_allow_html=True)

    #pl

    st.markdown('<div class="tab-title"><span class="tab-chip">FILE 03</span>Outcome Probability</div>', unsafe_allow_html=True)

    ledger_rows = [
        ("Dropout", probability_dict["Dropout"], "ink-crimson"),
        ("Enrolled", probability_dict["Enrolled"], "ink-brass"),
        ("Graduate", probability_dict["Graduate"], "ink-forest"),
    ]

    ledger_html = '<div class="ledger">'
    for row_i, (label_name, value, row_ink) in enumerate(ledger_rows):
        delay = 0.08 * row_i
        ledger_html += f'<div class="ledger-row" style="animation-delay:{delay}s"><div class="ledger-label">{label_name.upper()}</div><div class="ledger-track"><div class="ledger-fill {row_ink}" style="width:{value}%"></div></div><div class="ledger-value">{value:.1f}%</div></div>'
    ledger_html += '</div>'

    st.markdown(ledger_html, unsafe_allow_html=True)

   #shap exp
    st.markdown('<div class="tab-title"><span class="tab-chip">FILE 04</span>Why did the model make this prediction?</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-description">SHAP identifies which student characteristics contributed '
        'most strongly to the prediction.</div>',
        unsafe_allow_html=True
    )

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_df)

       #shap 3d array

        if isinstance(shap_values, list):
            class_shap = shap_values[prediction][0]
        elif isinstance(shap_values, np.ndarray):
            if shap_values.ndim == 3:
                class_shap = shap_values[0, :, prediction]
            elif shap_values.ndim == 2:
                class_shap = shap_values[0]
            else:
                class_shap = shap_values
        else:
            class_shap = np.array(shap_values).reshape(-1)

   #shap dataframe

        explanation = pd.DataFrame({
            "Feature": X.columns,
            "SHAP Value": class_shap
        })

        explanation["Absolute Impact"] = explanation["SHAP Value"].abs()
        explanation = explanation.sort_values("Absolute Impact", ascending=False)

        top_explanation = explanation.head(10)

        # ----------------------------------------------------
        # DISPLAY TABLE
        # ----------------------------------------------------

        display_explanation = top_explanation[["Feature", "SHAP Value"]].copy()
        display_explanation["SHAP Value"] = display_explanation["SHAP Value"].round(4)

        st.dataframe(
            display_explanation,
            use_container_width=True,
            hide_index=True
        )

        # ----------------------------------------------------
        # SHAP BAR CHART — recolored to match the glass palette
        # ----------------------------------------------------

        plt.rcParams["font.family"] = "DejaVu Sans"

        fig, ax = plt.subplots(figsize=(10, 6))
        fig.patch.set_facecolor("#2a1a4f"); fig.patch.set_alpha(1.0)
        ax.set_facecolor("#2c1c50")

        plot_data = top_explanation.iloc[::-1]
        bar_colors = ["#3fe8b0" if v > 0 else "#ff5d7a" for v in plot_data["SHAP Value"]]

        ax.barh(plot_data["Feature"], plot_data["SHAP Value"], color=bar_colors)
        ax.set_xlabel("SHAP Impact", color="#e6d9f7")
        ax.set_ylabel("Feature", color="#e6d9f7")
        ax.set_title(f"Top Factors Influencing {predicted_label} Prediction", color="#fffbff")
        ax.tick_params(colors="#d8c6ee")
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        for spine in ["left", "bottom"]:
            ax.spines[spine].set_color("#e2a6cf")
        ax.axvline(0, color="#a98dd6", linewidth=1)

        fig.tight_layout()

        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

  #case notes

        st.markdown('<div class="tab-title" style="font-size:19px;margin-top:22px;">💡 Case Notes</div>', unsafe_allow_html=True)

        notes_html = '<div class="case-notes">'
        for _, row in top_explanation.head(5).iterrows():
            feature = row["Feature"]
            value = row["SHAP Value"]
            direction = "supported" if value > 0 else "worked against"
            notes_html += f'<div class="case-note-row">— <b>{feature}</b> {direction} the <b>{predicted_label}</b> prediction.</div>'
        notes_html += '</div>'

        st.markdown(notes_html, unsafe_allow_html=True)

    except Exception as e:
        st.warning("The prediction was successful, but SHAP explanation could not be generated.")
        st.exception(e)


#what if???
st.markdown("")
st.markdown('<div class="tab-title"><span class="tab-chip">FILE 05</span>🔍 What-If Explorer</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-description">See how dropout likelihood shifts as one factor changes, holding everything else at its current value.</div>',
    unsafe_allow_html=True
)

whatif_candidates = [
    f for f in [
        "Admission grade",
        "Previous qualification (grade)",
        "Curricular units 1st sem (grade)",
        "Curricular units 2nd sem (grade)",
        "Curricular units 1st sem (approved)",
        "Curricular units 2nd sem (approved)",
        "Age at enrollment",
        "Unemployment rate",
        "Inflation rate",
        "GDP"
    ] if f in X.columns
]

st.markdown('<div class="explorer-box">', unsafe_allow_html=True)

whatif_feature = st.selectbox(
    "Explore sensitivity to:",
    whatif_candidates,
    key="whatif_feature_select"
)

feature_min = float(X[whatif_feature].min())
feature_max = float(X[whatif_feature].max())
sweep_points = np.linspace(feature_min, feature_max, 25)

sweep_results = []
for point in sweep_points:
    trial_values = dict(student_values)
    trial_values[whatif_feature] = point
    _, _, _, sweep_dropout = run_prediction(trial_values, defaults)
    sweep_results.append(sweep_dropout)

current_value = student_values.get(whatif_feature, defaults[whatif_feature])

fig2, ax2 = plt.subplots(figsize=(10, 3.6))
fig2.patch.set_facecolor("#2a1a4f"); fig2.patch.set_alpha(1.0)
ax2.set_facecolor("#2c1c50")
ax2.plot(sweep_points, sweep_results, color="#e26fc4", linewidth=2.5)
ax2.fill_between(sweep_points, sweep_results, color="#8b7bf0", alpha=0.18)
ax2.scatter([current_value], [live_dropout], color="#ff5d7a", s=70, zorder=5, edgecolor="#fffbff", linewidth=1.5)
ax2.set_xlabel(whatif_feature, color="#e6d9f7")
ax2.set_ylabel("Dropout %", color="#e6d9f7")
ax2.tick_params(colors="#d8c6ee")
for spine in ["top", "right"]:
    ax2.spines[spine].set_visible(False)
for spine in ["left", "bottom"]:
    ax2.spines[spine].set_color("#e2a6cf")
fig2.tight_layout()

st.pyplot(fig2, use_container_width=True)
plt.close(fig2)

st.markdown('</div>', unsafe_allow_html=True)


#student compare

st.markdown("")
st.markdown('<div class="tab-title"><span class="tab-chip">FILE 06</span>⚖ Compare Two Students</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-description">Snapshot the current profile into Slot A or Slot B, adjust the form, then snapshot again to compare.</div>',
    unsafe_allow_html=True
)

save_col_a, save_col_b, clear_col = st.columns(3)

with save_col_a:
    if st.button("📌 Save current profile as Student A", use_container_width=True):
        st.session_state.profile_a = dict(student_values)

with save_col_b:
    if st.button("📌 Save current profile as Student B", use_container_width=True):
        st.session_state.profile_b = dict(student_values)

with clear_col:
    if st.button("✕ Clear comparison", use_container_width=True):
        st.session_state.profile_a = None
        st.session_state.profile_b = None

if st.session_state.profile_a is not None or st.session_state.profile_b is not None:

    compare_col_a, compare_col_b = st.columns(2)

    slot_dropout = {"A": None, "B": None}

    with compare_col_a:
        if st.session_state.profile_a is not None:
            _, cmp_label_a, _, cmp_dropout_a = run_prediction(st.session_state.profile_a, defaults)
            _, cmp_ink_a = risk_tier(cmp_dropout_a)
            slot_dropout["A"] = cmp_dropout_a
            st.markdown(f"""
<div class="compare-card"><div class="compare-name">Student A</div><div class="compare-outcome {cmp_ink_a}">{cmp_label_a}</div><div class="compare-sub">{cmp_dropout_a:.1f}% dropout likelihood</div></div>
""", unsafe_allow_html=True)
        else:
            st.markdown('<div class="compare-card"><div class="compare-name">Student A</div><div class="compare-sub">No snapshot saved yet</div></div>', unsafe_allow_html=True)

    with compare_col_b:
        if st.session_state.profile_b is not None:
            _, cmp_label_b, _, cmp_dropout_b = run_prediction(st.session_state.profile_b, defaults)
            _, cmp_ink_b = risk_tier(cmp_dropout_b)
            slot_dropout["B"] = cmp_dropout_b
            st.markdown(f"""
<div class="compare-card"><div class="compare-name">Student B</div><div class="compare-outcome {cmp_ink_b}">{cmp_label_b}</div><div class="compare-sub">{cmp_dropout_b:.1f}% dropout likelihood</div></div>
""", unsafe_allow_html=True)
        else:
            st.markdown('<div class="compare-card"><div class="compare-name">Student B</div><div class="compare-sub">No snapshot saved yet</div></div>', unsafe_allow_html=True)

    if slot_dropout["A"] is not None and slot_dropout["B"] is not None:
        delta = slot_dropout["A"] - slot_dropout["B"]
        if abs(delta) < 0.05:
            delta_text = "Student A and Student B carry essentially the same dropout likelihood."
        elif delta > 0:
            delta_text = f"Student A carries a {abs(delta):.1f} point higher dropout likelihood than Student B."
        else:
            delta_text = f"Student B carries a {abs(delta):.1f} point higher dropout likelihood than Student A."
        st.markdown(f'<div class="compare-delta">{delta_text}</div>', unsafe_allow_html=True)


#footer
st.markdown("""
<div class="footer">PREPARED BY AN AUTOMATED MODEL · FOR ADVISING USE ONLY · NOT AN OFFICIAL RECORD<br/>XGBOOST · SHAP · STREAMLIT</div>
""", unsafe_allow_html=True)