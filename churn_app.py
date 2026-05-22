import streamlit as st
import pandas as pd
import numpy as np
import joblib
import warnings
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import shap
import io
import base64
import hashlib
import json
import re
import time
import random
import tempfile
import os
from datetime import datetime, timedelta
from gtts import gTTS
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors as rl_colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                 Table, TableStyle, HRFlowable)
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ─── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ChurnShield AI · 3D Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)
# ─── Premium CSS with Glassmorphism + Animations ────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Space+Grotesk:wght@400;500;600;700&display=swap');

  html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background: #030712;
  }

  /* ── Animated background ── */
  .main {
    background: radial-gradient(ellipse at 20% 50%, rgba(124,58,237,0.08) 0%, transparent 60%),
                radial-gradient(ellipse at 80% 20%, rgba(167,139,250,0.06) 0%, transparent 50%),
                radial-gradient(ellipse at 50% 80%, rgba(52,211,153,0.04) 0%, transparent 50%),
                #030712;
  }
  .block-container { padding: 1.5rem 2.5rem 3rem; max-width: 1500px; }

  /* ── Hero Section ── */
  .hero-wrap {
    position: relative;
    background: linear-gradient(135deg, rgba(19,17,31,0.9) 0%, rgba(10,10,24,0.95) 100%);
    border: 1px solid rgba(124,58,237,0.3);
    border-radius: 24px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    overflow: hidden;
    backdrop-filter: blur(20px);
  }
  .hero-wrap::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: conic-gradient(from 0deg at 50% 50%,
      transparent 0deg, rgba(124,58,237,0.04) 60deg,
      transparent 120deg, rgba(34,211,238,0.03) 180deg,
      transparent 240deg, rgba(52,211,153,0.02) 300deg, transparent 360deg);
    animation: rotate 20s linear infinite;
  }
  @keyframes rotate { to { transform: rotate(360deg); } }

  .hero-badge {
    display: inline-flex; align-items: center; gap: .4rem;
    background: rgba(124,58,237,0.15);
    border: 1px solid rgba(124,58,237,0.4);
    border-radius: 20px;
    padding: .3rem .9rem;
    font-size: .75rem; font-weight: 600;
    color: #22d3ee; letter-spacing: .08em;
    text-transform: uppercase; margin-bottom: 1rem;
  }
  .hero-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.6rem; font-weight: 700;
    background: linear-gradient(135deg, #f1f0ff 0%, #a78bfa 40%, #22d3ee 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; margin: 0 0 .5rem;
    line-height: 1.2;
  }
  .hero-sub { font-size: .95rem; color: #9d9abf; max-width: 600px; line-height: 1.6; }
  .hero-stats {
    display: flex; gap: 2rem; margin-top: 1.5rem; flex-wrap: wrap;
  }
  .hero-stat { text-align: center; }
  .hero-stat-val {
    font-size: 1.6rem; font-weight: 700;
    background: linear-gradient(135deg, #a78bfa, #22d3ee);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text;
    font-family: 'Space Grotesk', sans-serif;
  }
  .hero-stat-lbl { font-size: .72rem; color: #9d9abf; text-transform: uppercase; letter-spacing: .08em; }

  /* ── Glass KPI Cards ── */
  .glass-card {
    background: rgba(19,17,31,0.6);
    border: 1px solid rgba(124,58,237,0.2);
    border-radius: 16px;
    padding: 1.5rem 1.8rem;
    backdrop-filter: blur(20px);
    transition: all .3s ease;
    position: relative; overflow: hidden;
  }
  .glass-card::after {
    content: '';
    position: absolute; top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(34,211,238,0.4), transparent);
  }
  .glass-card:hover {
    border-color: rgba(34,211,238,0.45);
    transform: translateY(-3px);
    box-shadow: 0 20px 40px rgba(124,58,237,0.12), 0 0 20px rgba(34,211,238,0.08);
  }
  .kpi-icon { font-size: 1.8rem; margin-bottom: .6rem; }
  .kpi-value {
    font-size: 2.4rem; font-weight: 800;
    background: linear-gradient(135deg, #a78bfa, #22d3ee);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text;
    font-family: 'Space Grotesk', sans-serif; line-height: 1;
  }
  .kpi-label { font-size: .75rem; color: #9d9abf; text-transform: uppercase; letter-spacing: .1em; margin-top: .4rem; }
  .kpi-delta { font-size: .82rem; margin-top: .5rem; display: flex; align-items: center; gap: .3rem; }

  /* ── Section Headers ── */
  .section-header {
    display: flex; align-items: center; gap: .8rem;
    margin-bottom: 1.2rem; padding-bottom: .7rem;
    border-bottom: 1px solid rgba(45,43,69,0.8);
  }
  .section-icon { font-size: 1.2rem; }
  .section-title {
    font-size: 1.05rem; font-weight: 600; color: #d4d0f0;
    font-family: 'Space Grotesk', sans-serif;
  }

  /* ── Predict Button ── */
  div.stButton > button {
    background: linear-gradient(135deg, #7c3aed 0%, #a78bfa 60%, #22d3ee 100%);
    color: #fff; border: none; border-radius: 12px;
    padding: .85rem 2.5rem; font-size: 1rem; font-weight: 700;
    width: 100%; letter-spacing: .03em;
    transition: all .3s ease; cursor: pointer;
    box-shadow: 0 4px 20px rgba(124,58,237,0.35);
    font-family: 'Space Grotesk', sans-serif;
  }
  div.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 30px rgba(34,211,238,0.4);
  }
  div.stButton > button:active { transform: translateY(0); }

  /* ── Result Cards ── */
  .result-card {
    border-radius: 20px; padding: 2rem 2.2rem;
    text-align: center; position: relative; overflow: hidden;
  }
  .result-safe {
    background: linear-gradient(135deg, rgba(6,46,36,0.8), rgba(6,36,46,0.8));
    border: 2px solid rgba(52,211,153,0.5);
    box-shadow: 0 0 40px rgba(52,211,153,0.1), inset 0 0 40px rgba(52,211,153,0.03);
  }
  .result-risk {
    background: linear-gradient(135deg, rgba(46,6,6,0.8), rgba(46,20,6,0.8));
    border: 2px solid rgba(248,113,113,0.5);
    box-shadow: 0 0 40px rgba(248,113,113,0.1), inset 0 0 40px rgba(248,113,113,0.03);
  }
  .result-label { font-size: 1.3rem; font-weight: 700; margin-bottom: .5rem; font-family: 'Space Grotesk', sans-serif; }
  .result-prob { font-size: 3.5rem; font-weight: 800; font-family: 'Space Grotesk', sans-serif; line-height: 1; }
  .result-desc { font-size: .85rem; color: #9d9abf; margin-top: .8rem; line-height: 1.5; }

  /* ── Risk Tier Badge ── */
  .tier-badge {
    display: inline-flex; align-items: center; gap: .5rem;
    border-radius: 25px; padding: .5rem 1.2rem;
    font-size: .88rem; font-weight: 600; margin-top: 1rem;
    font-family: 'Space Grotesk', sans-serif;
  }
  .tier-critical { background: rgba(248,113,113,0.15); border: 1px solid rgba(248,113,113,0.4); color: #f87171; }
  .tier-high     { background: rgba(251,191,36,0.15); border: 1px solid rgba(251,191,36,0.4); color: #fbbf24; }
  .tier-medium   { background: rgba(253,230,138,0.15);  border: 1px solid rgba(253,230,138,0.4);  color: #fde68a; }
  .tier-low      { background: rgba(52,211,153,0.15);  border: 1px solid rgba(52,211,153,0.4);  color: #34d399; }

  /* ── Factor Pills ── */
  .pill-risk {
    display: inline-flex; align-items: center; gap: .3rem;
    background: rgba(248,113,113,0.1); color: #f87171;
    border: 1px solid rgba(248,113,113,0.3);
    border-radius: 20px; padding: .3rem .9rem;
    font-size: .8rem; margin: .2rem;
  }
  .pill-safe {
    display: inline-flex; align-items: center; gap: .3rem;
    background: rgba(52,211,153,0.1); color: #34d399;
    border: 1px solid rgba(52,211,153,0.3);
    border-radius: 20px; padding: .3rem .9rem;
    font-size: .8rem; margin: .2rem;
  }

  /* ── Advice Box ── */
  .advice-glass {
    background: rgba(19,17,31,0.7);
    border: 1px solid rgba(34,211,238,0.25);
    border-radius: 14px; padding: 1.2rem 1.4rem;
    font-size: .88rem; color: #d4d0f0; margin-top: 1rem;
    backdrop-filter: blur(10px);
  }
  .advice-glass .advice-title {
    color: #22d3ee; font-weight: 700; font-size: .9rem;
    margin-bottom: .8rem; display: flex; align-items: center; gap: .4rem;
    font-family: 'Space Grotesk', sans-serif;
  }
  .advice-item {
    display: flex; align-items: flex-start; gap: .6rem;
    padding: .4rem 0; border-bottom: 1px solid rgba(45,43,69,0.5);
    line-height: 1.5;
  }
  .advice-item:last-child { border-bottom: none; }
  .advice-dot { color: #22d3ee; font-size: 1rem; flex-shrink: 0; margin-top: .1rem; }

  /* ── Sidebar ── */
  [data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0a0a18 0%, #0f0f1a 100%) !important;
    border-right: 1px solid rgba(45,43,69,0.6);
  }
  .sidebar-section-header {
    font-size: .72rem; font-weight: 700; color: #22d3ee;
    text-transform: uppercase; letter-spacing: .12em;
    margin: 1.2rem 0 .6rem; display: flex; align-items: center; gap: .4rem;
  }
  .sidebar-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(34,211,238,0.25), transparent);
    margin: 1rem 0;
  }

  /* ── Tabs ── */
  .stTabs [data-baseweb="tab-list"] {
    background: rgba(19,17,31,0.6); border-radius: 12px;
    padding: 4px; gap: 2px; border: 1px solid rgba(45,43,69,0.5);
  }
  .stTabs [data-baseweb="tab"] {
    border-radius: 9px !important; padding: .55rem 1.4rem;
    color: #9d9abf; font-weight: 500; font-size: .9rem;
    transition: all .2s;
  }
  .stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #7c3aed, #22d3ee) !important;
    color: #fff !important; font-weight: 600;
    box-shadow: 0 4px 15px rgba(34,211,238,0.3);
  }

  /* ── Placeholder ── */
  .placeholder-wrap {
    background: rgba(19,17,31,0.4);
    border: 1px dashed rgba(124,58,237,0.3);
    border-radius: 20px; padding: 3.5rem 2rem;
    text-align: center; margin-top: 1rem;
  }
  .placeholder-icon { font-size: 4rem; margin-bottom: 1rem; }
  .placeholder-title {
    font-size: 1.4rem; font-weight: 700; color: #d4d0f0;
    font-family: 'Space Grotesk', sans-serif; margin-bottom: .6rem;
  }
  .placeholder-sub { color: #9d9abf; font-size: .92rem; max-width: 480px; margin: auto; line-height: 1.6; }
  .feature-grid {
    display: flex; justify-content: center; gap: 2.5rem;
    flex-wrap: wrap; margin-top: 2.5rem;
  }
  .feature-item { text-align: center; }
  .feature-item-icon { font-size: 1.8rem; }
  .feature-item-lbl { color: #9d9abf; font-size: .78rem; margin-top: .3rem; }

  /* ── 3D Chart Container ── */
  .chart-3d-wrap {
    background: rgba(3,7,18,0.8);
    border: 1px solid rgba(45,43,69,0.6);
    border-radius: 16px; padding: .5rem;
    backdrop-filter: blur(10px);
  }

  /* ── Scrollbar ── */
  ::-webkit-scrollbar { width: 6px; }
  ::-webkit-scrollbar-track { background: #0f0f1a; }
  ::-webkit-scrollbar-thumb { background: #2d2b45; border-radius: 3px; }
  ::-webkit-scrollbar-thumb:hover { background: #7c3aed; }

  /* ── Login Card ── */
  .login-wrap {
    max-width: 420px; margin: 4rem auto;
    background: rgba(19,17,31,0.85);
    border: 1px solid rgba(124,58,237,0.35);
    border-radius: 24px; padding: 2.5rem 2.8rem;
    backdrop-filter: blur(20px);
    box-shadow: 0 30px 60px rgba(0,0,0,0.5), 0 0 40px rgba(124,58,237,0.1);
  }
  .login-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.8rem; font-weight: 700; text-align: center;
    background: linear-gradient(135deg, #a78bfa, #22d3ee);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; margin-bottom: .4rem;
  }
  .login-sub { text-align: center; color: #9d9abf; font-size: .88rem; margin-bottom: 1.8rem; }

  /* ── Segment Cards ── */
  .seg-card {
    border-radius: 14px; padding: 1.2rem 1.4rem;
    border: 1px solid; margin-bottom: .5rem;
    transition: transform .2s;
  }
  .seg-card:hover { transform: translateY(-2px); }
  .seg-title { font-family: 'Space Grotesk', sans-serif; font-weight: 700; font-size: 1rem; }
  .seg-desc  { font-size: .82rem; color: #9d9abf; margin-top: .3rem; line-height: 1.5; }

  /* ── Why Churn Section ── */
  .why-card {
    background: rgba(19,17,31,0.6);
    border-radius: 14px; padding: 1.3rem 1.5rem;
    border-left: 4px solid;
    margin-bottom: .8rem;
    transition: all .2s;
  }
  .why-card:hover { transform: translateX(4px); }
  .why-title { font-family: 'Space Grotesk', sans-serif; font-weight: 700; font-size: .95rem; margin-bottom: .4rem; }
  .why-desc  { font-size: .83rem; color: #9d9abf; line-height: 1.6; }

  /* ── Sim pulse ── */
  @keyframes pulse-dot {
    0%,100% { opacity:1; transform:scale(1); }
    50%      { opacity:.5; transform:scale(1.4); }
  }
  .live-dot {
    display:inline-block; width:8px; height:8px;
    background:#34d399; border-radius:50%;
    animation: pulse-dot 1.5s ease-in-out infinite;
    margin-right:6px;
  }

  /* ── Alert badge ── */
  .alert-item {
    display:flex; align-items:flex-start; gap:.8rem;
    background:rgba(248,113,113,0.07);
    border:1px solid rgba(248,113,113,0.2);
    border-radius:10px; padding:.8rem 1rem; margin-bottom:.5rem;
  }
  .alert-icon { font-size:1.2rem; flex-shrink:0; }
  .alert-body { font-size:.84rem; color:#d4d0f0; line-height:1.5; }
  .alert-time { font-size:.72rem; color:#9d9abf; margin-top:.2rem; }
</style>
""", unsafe_allow_html=True)

# ─── Session State Init ──────────────────────────────────────────────────────────
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False
if "username" not in st.session_state:
    st.session_state["username"] = ""
if "run_prediction" not in st.session_state:
    st.session_state["run_prediction"] = True


# ─── Load Assets ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load("Logistic_Model.pkl")

@st.cache_data
def load_data():
    return pd.read_csv("customer_churn_prediction_dataset.csv")

model = load_model()
df    = load_data()
FEATURE_COLS = list(model.feature_names_in_)


# ─── Encode customer ────────────────────────────────────────────────────────────
def encode_customer(inputs: dict) -> pd.DataFrame:
    row = {f: 0 for f in FEATURE_COLS}
    row["SeniorCitizen"]   = inputs["SeniorCitizen"]
    row["tenure"]          = inputs["tenure"]
    row["MonthlyCharges"]  = inputs["MonthlyCharges"]
    row["TotalCharges"]    = inputs["TotalCharges"]
    oh_map = {
        "gender": inputs["gender"], "Partner": inputs["Partner"],
        "Dependents": inputs["Dependents"], "PhoneService": inputs["PhoneService"],
        "MultipleLines": inputs["MultipleLines"], "InternetService": inputs["InternetService"],
        "OnlineSecurity": inputs["OnlineSecurity"], "OnlineBackup": inputs["OnlineBackup"],
        "DeviceProtection": inputs["DeviceProtection"], "TechSupport": inputs["TechSupport"],
        "StreamingTV": inputs["StreamingTV"], "StreamingMovies": inputs["StreamingMovies"],
        "Contract": inputs["Contract"], "PaperlessBilling": inputs["PaperlessBilling"],
        "PaymentMethod": inputs["PaymentMethod"],
    }
    for col, val in oh_map.items():
        key = f"{col}_{val}"
        if key in row:
            row[key] = 1
    return pd.DataFrame([row])[FEATURE_COLS]


# ─── 3D Gauge Chart ─────────────────────────────────────────────────────────────
def gauge_3d(prob: float):
    color = "#f87171" if prob >= .6 else "#fbbf24" if prob >= .35 else "#22d3ee"
    glow  = "rgba(248,113,113,0.3)" if prob >= .6 else "rgba(251,191,36,0.3)" if prob >= .35 else "rgba(52,211,153,0.3)"
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=round(prob * 100, 1),
        delta={"reference": 50, "valueformat": ".1f",
               "increasing": {"color": "#f87171"}, "decreasing": {"color": "#34d399"}},
        number={"suffix": "%", "font": {"size": 42, "color": "#f1f0ff", "family": "Space Grotesk"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#4b4869",
                     "tickfont": {"color": "#9d9abf", "size": 10}, "tickwidth": 1,
                     "tickvals": [0, 25, 50, 75, 100]},
            "bar":  {"color": color, "thickness": .22},
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 0,
            "steps": [
                {"range": [0,  25], "color": "rgba(52,211,153,0.08)"},
                {"range": [25, 50], "color": "rgba(52,211,153,0.04)"},
                {"range": [50, 75], "color": "rgba(251,191,36,0.06)"},
                {"range": [75,100], "color": "rgba(248,113,113,0.08)"},
            ],
            "threshold": {
                "line": {"color": color, "width": 3},
                "thickness": .8, "value": round(prob*100, 1)
            },
        },
        title={"text": "Churn Probability", "font": {"color": "#9d9abf", "size": 13, "family": "Inter"}},
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=70, b=10, l=30, r=30), height=260,
        font={"family": "Inter"},
        annotations=[dict(
            x=0.5, y=0.18, text=f"<b style='color:{color}'>{'HIGH RISK' if prob>=.5 else 'LOW RISK'}</b>",
            showarrow=False, font=dict(size=11, color=color, family="Space Grotesk"),
            xref="paper", yref="paper"
        )]
    )
    return fig


# ─── 3D Feature Impact Chart ────────────────────────────────────────────────────
def feature_impact_3d(X_row: pd.DataFrame):
    coef    = model.coef_[0]
    values  = X_row.values[0]
    contrib = coef * values
    top_idx = np.argsort(np.abs(contrib))[-12:]
    names   = [FEATURE_COLS[i].replace("_", " ") for i in top_idx]
    vals    = [contrib[i] for i in top_idx]
    colors  = ["#f87171" if v > 0 else "#22d3ee" for v in vals]
    sizes   = [abs(v) * 300 + 8 for v in vals]

    fig = go.Figure()
    # 3D bar-like effect using scatter3d
    for i, (name, val, col) in enumerate(zip(names, vals, colors)):
        fig.add_trace(go.Scatter3d(
            x=[i, i], y=[0, val], z=[0, 0],
            mode="lines+markers",
            line=dict(color=col, width=6),
            marker=dict(size=[2, 10], color=col,
                        symbol=["circle", "circle"],
                        line=dict(color="white", width=1)),
            name=name,
            hovertemplate=f"<b>{name}</b><br>Impact: {val:.4f}<extra></extra>",
            showlegend=False,
        ))

    fig.update_layout(
        scene=dict(
            xaxis=dict(
                tickvals=list(range(len(names))),
                ticktext=[n[:18] for n in names],
                tickfont=dict(color="#9d9abf", size=9),
                gridcolor="rgba(45,43,69,0.5)",
                backgroundcolor="rgba(0,0,0,0)",
                showbackground=True,
                zerolinecolor="#2d2b45",
            ),
            yaxis=dict(
                title=dict(text="Impact Score", font=dict(color="#9d9abf", size=11)),
                tickfont=dict(color="#9d9abf", size=9),
                gridcolor="rgba(45,43,69,0.5)",
                backgroundcolor="rgba(0,0,0,0)",
                showbackground=True,
                zerolinecolor="#7c3aed",
            ),
            zaxis=dict(
                showticklabels=False, showgrid=False,
                backgroundcolor="rgba(0,0,0,0)",
            ),
            bgcolor="rgba(0,0,0,0)",
            camera=dict(eye=dict(x=1.8, y=1.2, z=0.8)),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=10, b=10, l=0, r=0),
        height=380,
        font={"family": "Inter", "color": "#9d9abf"},
    )
    return fig


# ─── 3D Scatter: Tenure vs Monthly vs Churn ─────────────────────────────────────
def scatter_3d_churn():
    colors = df["Churn"].map({"No": "#7c3aed", "Yes": "#f87171"})
    sizes  = (df["TotalCharges"].fillna(0) / df["TotalCharges"].max() * 10 + 4).tolist()

    fig = go.Figure()
    for churn_val, color, label in [("No", "#7c3aed", "Retained"), ("Yes", "#f87171", "Churned")]:
        mask = df["Churn"] == churn_val
        fig.add_trace(go.Scatter3d(
            x=df[mask]["tenure"],
            y=df[mask]["MonthlyCharges"],
            z=df[mask]["TotalCharges"].fillna(0),
            mode="markers",
            name=label,
            marker=dict(
                size=5,
                color=color,
                opacity=0.75,
                line=dict(color="rgba(255,255,255,0.1)", width=0.5),
            ),
            hovertemplate=(
                "<b>" + label + "</b><br>"
                "Tenure: %{x} mo<br>"
                "Monthly: $%{y:.0f}<br>"
                "Total: $%{z:.0f}<extra></extra>"
            ),
        ))

    fig.update_layout(
        scene=dict(
            xaxis=dict(title=dict(text="Tenure (months)", font=dict(color="#9d9abf", size=11)),
                       tickfont=dict(color="#9d9abf", size=9),
                       gridcolor="rgba(45,43,69,0.6)", backgroundcolor="rgba(3,7,18,0.8)",
                       showbackground=True),
            yaxis=dict(title=dict(text="Monthly Charges ($)", font=dict(color="#9d9abf", size=11)),
                       tickfont=dict(color="#9d9abf", size=9),
                       gridcolor="rgba(45,43,69,0.6)", backgroundcolor="rgba(3,7,18,0.8)",
                       showbackground=True),
            zaxis=dict(title=dict(text="Total Charges ($)", font=dict(color="#9d9abf", size=11)),
                       tickfont=dict(color="#9d9abf", size=9),
                       gridcolor="rgba(45,43,69,0.6)", backgroundcolor="rgba(3,7,18,0.8)",
                       showbackground=True),
            bgcolor="rgba(3,7,18,0.9)",
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.0)),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(
            bgcolor="rgba(19,17,31,0.8)", bordercolor="rgba(124,58,237,0.3)",
            borderwidth=1, font=dict(color="#d4d0f0", size=11),
            x=0.02, y=0.98,
        ),
        margin=dict(t=10, b=10, l=0, r=0),
        height=480,
        font={"family": "Inter"},
    )
    return fig


# ─── 3D Surface: Risk Landscape ─────────────────────────────────────────────────
def risk_surface_3d():
    tenure_range  = np.linspace(1, 72, 40)
    monthly_range = np.linspace(18, 120, 40)
    T, M = np.meshgrid(tenure_range, monthly_range)

    # Approximate risk surface using logistic-like formula
    risk = 1 / (1 + np.exp(0.04 * T - 0.015 * M + 0.5))

    fig = go.Figure(go.Surface(
        x=T, y=M, z=risk,
        colorscale=[
            [0.0,  "#22d3ee"],
            [0.3,  "#a78bfa"],
            [0.65, "#fbbf24"],
            [1.0,  "#f87171"],
        ],
        opacity=0.85,
        contours=dict(
            z=dict(show=True, usecolormap=True, highlightcolor="#a78bfa", project_z=True)
        ),
        hovertemplate="Tenure: %{x:.0f} mo<br>Monthly: $%{y:.0f}<br>Risk: %{z:.1%}<extra></extra>",
        showscale=True,
        colorbar=dict(
            title=dict(text="Churn Risk", font=dict(color="#9d9abf", size=11)),
            tickfont=dict(color="#9d9abf", size=9),
            bgcolor="rgba(19,17,31,0.8)",
            bordercolor="rgba(124,58,237,0.3)",
            borderwidth=1,
            tickformat=".0%",
        ),
    ))
    fig.update_layout(
        scene=dict(
            xaxis=dict(title=dict(text="Tenure (months)", font=dict(color="#9d9abf", size=11)),
                       tickfont=dict(color="#9d9abf", size=9),
                       gridcolor="rgba(45,43,69,0.6)", backgroundcolor="rgba(3,7,18,0.8)",
                       showbackground=True),
            yaxis=dict(title=dict(text="Monthly Charges ($)", font=dict(color="#9d9abf", size=11)),
                       tickfont=dict(color="#9d9abf", size=9),
                       gridcolor="rgba(45,43,69,0.6)", backgroundcolor="rgba(3,7,18,0.8)",
                       showbackground=True),
            zaxis=dict(title=dict(text="Churn Risk", font=dict(color="#9d9abf", size=11)),
                       tickfont=dict(color="#9d9abf", size=9),
                       gridcolor="rgba(45,43,69,0.6)", backgroundcolor="rgba(3,7,18,0.8)",
                       showbackground=True, tickformat=".0%"),
            bgcolor="rgba(3,7,18,0.9)",
            camera=dict(eye=dict(x=1.6, y=1.6, z=1.2)),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=10, b=10, l=0, r=0),
        height=480,
        font={"family": "Inter"},
    )
    return fig


# ─── 3D Bar: Churn by Contract & Internet ───────────────────────────────────────
def bar_3d_churn():
    contracts = df["Contract"].unique().tolist()
    internets = df["InternetService"].unique().tolist()

    x_vals, y_vals, z_vals, colors_list = [], [], [], []
    for i, contract in enumerate(contracts):
        for j, internet in enumerate(internets):
            mask = (df["Contract"] == contract) & (df["InternetService"] == internet)
            subset = df[mask]
            if len(subset) > 0:
                churn_rate = (subset["Churn"] == "Yes").mean()
                x_vals.append(i)
                y_vals.append(j)
                z_vals.append(churn_rate)
                colors_list.append(churn_rate)

    fig = go.Figure(go.Scatter3d(
        x=x_vals, y=y_vals, z=z_vals,
        mode="markers",
        marker=dict(
            size=[z * 30 + 8 for z in z_vals],
            color=colors_list,
            colorscale=[[0, "#22d3ee"], [0.5, "#a78bfa"], [1, "#f87171"]],
            opacity=0.85,
            colorbar=dict(
                title=dict(text="Churn Rate", font=dict(color="#9d9abf", size=11)),
                tickfont=dict(color="#9d9abf", size=9),
                tickformat=".0%",
                bgcolor="rgba(19,17,31,0.8)",
                bordercolor="rgba(124,58,237,0.3)",
                borderwidth=1,
            ),
            line=dict(color="rgba(255,255,255,0.2)", width=1),
        ),
        hovertemplate=(
            "Contract: %{customdata[0]}<br>"
            "Internet: %{customdata[1]}<br>"
            "Churn Rate: %{z:.1%}<extra></extra>"
        ),
        customdata=[[contracts[int(x)], internets[int(y)]] for x, y in zip(x_vals, y_vals)],
    ))
    fig.update_layout(
        scene=dict(
            xaxis=dict(
                tickvals=list(range(len(contracts))),
                ticktext=contracts,
                title=dict(text="Contract Type", font=dict(color="#9d9abf", size=11)),
                tickfont=dict(color="#9d9abf", size=9),
                gridcolor="rgba(45,43,69,0.6)", backgroundcolor="rgba(3,7,18,0.8)",
                showbackground=True,
            ),
            yaxis=dict(
                tickvals=list(range(len(internets))),
                ticktext=internets,
                title=dict(text="Internet Service", font=dict(color="#9d9abf", size=11)),
                tickfont=dict(color="#9d9abf", size=9),
                gridcolor="rgba(45,43,69,0.6)", backgroundcolor="rgba(3,7,18,0.8)",
                showbackground=True,
            ),
            zaxis=dict(
                title=dict(text="Churn Rate", font=dict(color="#9d9abf", size=11)),
                tickfont=dict(color="#9d9abf", size=9),
                gridcolor="rgba(45,43,69,0.6)", backgroundcolor="rgba(3,7,18,0.8)",
                showbackground=True, tickformat=".0%",
            ),
            bgcolor="rgba(3,7,18,0.9)",
            camera=dict(eye=dict(x=1.8, y=1.4, z=1.2)),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=10, b=10, l=0, r=0),
        height=460,
        font={"family": "Inter"},
    )
    return fig


# ─── 2D helper charts ────────────────────────────────────────────────────────────
def churn_donut():
    counts = df["Churn"].value_counts()
    fig = go.Figure(go.Pie(
        labels=["Retained", "Churned"],
        values=[counts.get("No", 0), counts.get("Yes", 0)],
        hole=.62,
        marker=dict(colors=["#22d3ee", "#f87171"],
                    line=dict(color="#030712", width=3)),
        textinfo="label+percent",
        hoverinfo="label+value",
        textfont=dict(color="#d4d0f0", size=12),
    ))
    fig.add_annotation(
        text=f"<b style='font-size:22px;color:#f1f0ff'>{counts.get('Yes',0)/len(df):.0%}</b><br>"
             "<span style='font-size:11px;color:#9d9abf'>Churn Rate</span>",
        x=0.5, y=0.5, showarrow=False, align="center",
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", showlegend=True,
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#d4d0f0", size=11)),
        margin=dict(t=20, b=10, l=0, r=0), height=260,
        font={"family": "Inter"},
    )
    return fig


def tenure_hist():
    fig = px.histogram(
        df, x="tenure", color="Churn",
        color_discrete_map={"No": "#22d3ee", "Yes": "#f87171"},
        nbins=24, barmode="overlay", opacity=.8,
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#9d9abf", "family": "Inter"},
        xaxis=dict(gridcolor="rgba(45,43,69,0.5)", title="Tenure (months)"),
        yaxis=dict(gridcolor="rgba(45,43,69,0.5)", title="Count"),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#d4d0f0")),
        margin=dict(t=20, b=40, l=40, r=10), height=260,
    )
    return fig


def churn_by_col(col: str, label: str):
    grp = df.groupby(col)["Churn"].value_counts(normalize=True).unstack().fillna(0) * 100
    fig = go.Figure()
    fig.add_bar(name="Retained", x=grp.index.astype(str),
                y=grp.get("No", pd.Series(0, index=grp.index)),
                marker_color="#22d3ee", marker_line_color="rgba(0,0,0,0)")
    fig.add_bar(name="Churned",  x=grp.index.astype(str),
                y=grp.get("Yes", pd.Series(0, index=grp.index)),
                marker_color="#f87171", marker_line_color="rgba(0,0,0,0)")
    fig.update_layout(
        barmode="stack",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#9d9abf", "family": "Inter"},
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#d4d0f0")),
        xaxis=dict(gridcolor="rgba(45,43,69,0.5)", title=label),
        yaxis=dict(gridcolor="rgba(45,43,69,0.5)", title="% Customers"),
        margin=dict(t=20, b=40, l=40, r=20), height=260,
    )
    return fig


def monthly_violin():
    fill_map = {"#22d3ee": "rgba(34,211,238,0.15)", "#f87171": "rgba(248,113,113,0.15)"}
    fig = go.Figure()
    for churn_val, color, label in [("No", "#22d3ee", "Retained"), ("Yes", "#f87171", "Churned")]:
        fig.add_trace(go.Violin(
            y=df[df["Churn"] == churn_val]["MonthlyCharges"],
            name=label, line_color=color,
            fillcolor=fill_map[color],
            box_visible=True, meanline_visible=True,
            points="outliers",
        ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#9d9abf", "family": "Inter"},
        yaxis=dict(gridcolor="rgba(45,43,69,0.5)", title="Monthly Charges ($)"),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#d4d0f0")),
        margin=dict(t=20, b=20, l=40, r=10), height=260,
        violingap=0.3,
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#                         NEW FEATURE HELPERS
# ══════════════════════════════════════════════════════════════════════════════

# ─── SHAP Explainability ────────────────────────────────────────────────────────
@st.cache_resource
def get_shap_explainer(_model, _X_background):
    """Build a LinearExplainer once and cache it."""
    return shap.LinearExplainer(_model, _X_background, feature_perturbation="interventional")

def shap_chart(X_row: pd.DataFrame, X_background: pd.DataFrame):
    explainer   = get_shap_explainer(model, X_background)
    shap_values = explainer.shap_values(X_row)[0]          # shape (n_features,)

    # Top 15 by absolute value
    top_idx  = np.argsort(np.abs(shap_values))[-15:]
    names    = [FEATURE_COLS[i].replace("_", " ") for i in top_idx]
    vals     = [shap_values[i] for i in top_idx]
    base_val = explainer.expected_value

    colors = ["#f87171" if v > 0 else "#22d3ee" for v in vals]

    fig = go.Figure()

    # Waterfall-style bars
    fig.add_trace(go.Bar(
        x=vals, y=names, orientation="h",
        marker=dict(
            color=colors,
            line=dict(color="rgba(255,255,255,0.05)", width=0.5),
        ),
        hovertemplate="<b>%{y}</b><br>SHAP value: %{x:.4f}<extra></extra>",
        name="SHAP Impact",
    ))

    # Base value line
    fig.add_vline(x=base_val, line_dash="dash",
                  line_color="rgba(167,139,250,0.6)", line_width=1.5,
                  annotation_text=f"Base: {base_val:.3f}",
                  annotation_font_color="#a78bfa",
                  annotation_font_size=10)

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#9d9abf", "family": "Inter"},
        xaxis=dict(gridcolor="rgba(45,43,69,0.5)", zerolinecolor="#a78bfa",
                   title="SHAP Value (impact on model output)"),
        yaxis=dict(gridcolor="rgba(0,0,0,0)"),
        margin=dict(t=10, b=10, l=10, r=20), height=420,
        showlegend=False,
    )
    return fig, shap_values


# ─── PDF Report Generator ───────────────────────────────────────────────────────
def generate_pdf_report(inputs: dict, prob: float, pred: int,
                         risk_factors: list, prot_factors: list,
                         advice: list, shap_vals: np.ndarray) -> bytes:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                             leftMargin=2*cm, rightMargin=2*cm,
                             topMargin=2*cm, bottomMargin=2*cm)

    styles = getSampleStyleSheet()
    # Custom styles
    title_style = ParagraphStyle("title", parent=styles["Title"],
                                  fontSize=22, textColor=rl_colors.HexColor("#a78bfa"),
                                  spaceAfter=4, fontName="Helvetica-Bold")
    sub_style   = ParagraphStyle("sub", parent=styles["Normal"],
                                  fontSize=10, textColor=rl_colors.HexColor("#9d9abf"),
                                  spaceAfter=12)
    h2_style    = ParagraphStyle("h2", parent=styles["Heading2"],
                                  fontSize=13, textColor=rl_colors.HexColor("#22d3ee"),
                                  spaceBefore=14, spaceAfter=6, fontName="Helvetica-Bold")
    body_style  = ParagraphStyle("body", parent=styles["Normal"],
                                  fontSize=10, textColor=rl_colors.HexColor("#1a1a2e"),
                                  spaceAfter=4, leading=14)
    risk_color  = "#f87171" if pred == 1 else "#34d399"
    result_txt  = "⚠ CHURN RISK DETECTED" if pred == 1 else "✓ LOW CHURN RISK"

    story = []

    # ── Header ──
    story.append(Paragraph("🛡️ ChurnShield AI", title_style))
    story.append(Paragraph("Customer Churn Prediction Report", sub_style))
    story.append(HRFlowable(width="100%", thickness=1,
                             color=rl_colors.HexColor("#a78bfa"), spaceAfter=12))

    # ── Result Banner ──
    result_style = ParagraphStyle("result", parent=styles["Normal"],
                                   fontSize=16, fontName="Helvetica-Bold",
                                   textColor=rl_colors.HexColor(risk_color),
                                   alignment=TA_CENTER, spaceAfter=4)
    prob_style   = ParagraphStyle("prob", parent=styles["Normal"],
                                   fontSize=28, fontName="Helvetica-Bold",
                                   textColor=rl_colors.HexColor(risk_color),
                                   alignment=TA_CENTER, spaceAfter=12)
    story.append(Paragraph(result_txt, result_style))
    story.append(Paragraph(f"Churn Probability: {prob:.1%}", prob_style))
    story.append(HRFlowable(width="100%", thickness=0.5,
                             color=rl_colors.HexColor("#cccccc"), spaceAfter=10))

    # ── Customer Profile ──
    story.append(Paragraph("Customer Profile", h2_style))
    profile_data = [
        ["Field", "Value", "Field", "Value"],
        ["Gender",          inputs["gender"],          "Senior Citizen",   "Yes" if inputs["SeniorCitizen"] else "No"],
        ["Partner",         inputs["Partner"],          "Dependents",       inputs["Dependents"]],
        ["Tenure",          f"{inputs['tenure']} months","Contract",        inputs["Contract"]],
        ["Monthly Charges", f"${inputs['MonthlyCharges']:.2f}", "Total Charges", f"${inputs['TotalCharges']:.2f}"],
        ["Internet Service",inputs["InternetService"],  "Phone Service",    inputs["PhoneService"]],
        ["Payment Method",  inputs["PaymentMethod"],    "Paperless Billing",inputs["PaperlessBilling"]],
        ["Online Security", inputs["OnlineSecurity"],   "Tech Support",     inputs["TechSupport"]],
        ["Streaming TV",    inputs["StreamingTV"],      "Streaming Movies", inputs["StreamingMovies"]],
    ]
    tbl = Table(profile_data, colWidths=[3.8*cm, 4.2*cm, 3.8*cm, 4.2*cm])
    tbl.setStyle(TableStyle([
        ("BACKGROUND",  (0,0), (-1,0), rl_colors.HexColor("#7c3aed")),
        ("TEXTCOLOR",   (0,0), (-1,0), rl_colors.white),
        ("FONTNAME",    (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE",    (0,0), (-1,0), 9),
        ("ROWBACKGROUNDS", (0,1), (-1,-1),
         [rl_colors.HexColor("#f5f3ff"), rl_colors.white]),
        ("FONTSIZE",    (0,1), (-1,-1), 9),
        ("GRID",        (0,0), (-1,-1), 0.4, rl_colors.HexColor("#dddddd")),
        ("FONTNAME",    (0,1), (0,-1), "Helvetica-Bold"),
        ("FONTNAME",    (2,1), (2,-1), "Helvetica-Bold"),
        ("VALIGN",      (0,0), (-1,-1), "MIDDLE"),
        ("TOPPADDING",  (0,0), (-1,-1), 5),
        ("BOTTOMPADDING",(0,0),(-1,-1), 5),
    ]))
    story.append(tbl)
    story.append(Spacer(1, 10))

    # ── Risk Factors ──
    story.append(Paragraph("Risk Analysis", h2_style))
    if risk_factors:
        story.append(Paragraph("<b>🔴 Churn-driving factors:</b>", body_style))
        for f in risk_factors:
            story.append(Paragraph(f"  • {f}", body_style))
    if prot_factors:
        story.append(Paragraph("<b>🟢 Retention-supporting factors:</b>", body_style))
        for f in prot_factors:
            story.append(Paragraph(f"  • {f}", body_style))
    story.append(Spacer(1, 6))

    # ── SHAP Top Features ──
    story.append(Paragraph("SHAP Feature Importance (Top 10)", h2_style))
    top_shap_idx = np.argsort(np.abs(shap_vals))[-10:][::-1]
    shap_data = [["Feature", "SHAP Value", "Direction"]]
    for i in top_shap_idx:
        direction = "↑ Increases Risk" if shap_vals[i] > 0 else "↓ Decreases Risk"
        shap_data.append([
            FEATURE_COLS[i].replace("_", " "),
            f"{shap_vals[i]:.4f}",
            direction,
        ])
    shap_tbl = Table(shap_data, colWidths=[8*cm, 3.5*cm, 4.5*cm])
    shap_tbl.setStyle(TableStyle([
        ("BACKGROUND",  (0,0), (-1,0), rl_colors.HexColor("#0e7490")),
        ("TEXTCOLOR",   (0,0), (-1,0), rl_colors.white),
        ("FONTNAME",    (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE",    (0,0), (-1,-1), 9),
        ("ROWBACKGROUNDS", (0,1), (-1,-1),
         [rl_colors.HexColor("#ecfeff"), rl_colors.white]),
        ("GRID",        (0,0), (-1,-1), 0.4, rl_colors.HexColor("#dddddd")),
        ("VALIGN",      (0,0), (-1,-1), "MIDDLE"),
        ("TOPPADDING",  (0,0), (-1,-1), 5),
        ("BOTTOMPADDING",(0,0),(-1,-1), 5),
    ]))
    story.append(shap_tbl)
    story.append(Spacer(1, 10))

    # ── Recommendations ──
    story.append(Paragraph("Recommended Actions", h2_style))
    for icon, text in advice:
        # Strip HTML tags for PDF
        clean = text.replace("<b style='color:#a78bfa'>","").replace("<b style='color:#22d3ee'>","").replace("</b>","")
        story.append(Paragraph(f"{icon}  {clean}", body_style))
    story.append(Spacer(1, 10))

    # ── Footer ──
    story.append(HRFlowable(width="100%", thickness=0.5,
                             color=rl_colors.HexColor("#cccccc"), spaceBefore=10))
    footer_style = ParagraphStyle("footer", parent=styles["Normal"],
                                   fontSize=8, textColor=rl_colors.HexColor("#9d9abf"),
                                   alignment=TA_CENTER)
    story.append(Paragraph("Generated by ChurnShield AI · Powered by Logistic Regression + SHAP", footer_style))

    doc.build(story)
    return buf.getvalue()


# ─── AI Voice Assistant ─────────────────────────────────────────────────────────
def generate_voice_summary(prob: float, pred: int, risk_factors: list,
                             prot_factors: list, advice: list,
                             tenure: int, contract: str, monthly: float) -> bytes:
    tier = ("critical" if prob >= .75 else "high" if prob >= .5
            else "medium" if prob >= .3 else "low")

    risk_txt = (", ".join(risk_factors[:3]) if risk_factors
                else "no significant risk factors detected")
    prot_txt = (", ".join(prot_factors[:2]) if prot_factors
                else "no strong protective factors")

    # Build clean advice text
    advice_clean = []
    for _, text in advice[:2]:
        clean = (text.replace("<b style='color:#a78bfa'>","")
                     .replace("<b style='color:#22d3ee'>","")
                     .replace("</b>",""))
        advice_clean.append(clean)

    if pred == 1:
        summary = (
            f"Attention! This customer has a {prob:.0%} churn probability, "
            f"placing them in the {tier} risk tier. "
            f"The main churn-driving factors are: {risk_txt}. "
            f"Protective factors include: {prot_txt}. "
            f"The customer has been with us for {tenure} months "
            f"on a {contract} contract, paying ${monthly:.0f} per month. "
            f"Recommended actions: {'. '.join(advice_clean)}. "
            f"Immediate retention intervention is advised."
        )
    else:
        summary = (
            f"Good news! This customer has only a {prob:.0%} churn probability, "
            f"placing them in the {tier} risk tier. "
            f"They appear to be a loyal customer with {tenure} months of tenure "
            f"on a {contract} contract. "
            f"Protective factors include: {prot_txt}. "
            f"Consider offering a loyalty reward or upsell opportunity."
        )

    tts = gTTS(text=summary, lang="en", slow=False)
    audio_buf = io.BytesIO()
    tts.write_to_fp(audio_buf)
    audio_buf.seek(0)
    return audio_buf.read()


# ══════════════════════════════════════════════════════════════════════════════
#                     NEW FEATURE HELPERS
# ══════════════════════════════════════════════════════════════════════════════

# ─── Login System ───────────────────────────────────────────────────────────────
USERS = {
    "admin":   hashlib.sha256("admin123".encode()).hexdigest(),
    "analyst": hashlib.sha256("analyst2024".encode()).hexdigest(),
    "demo":    hashlib.sha256("demo".encode()).hexdigest(),
}

def check_login(username: str, password: str) -> bool:
    return USERS.get(username) == hashlib.sha256(password.encode()).hexdigest()

def login_page():
    st.markdown("""
    <div style="display:flex;align-items:center;justify-content:center;min-height:60vh">
    <div class="login-wrap">
      <div style="text-align:center;font-size:3rem;margin-bottom:.5rem">🛡️</div>
      <div class="login-title">ChurnShield AI</div>
      <div class="login-sub">Sign in to access the intelligence platform</div>
    </div></div>
    """, unsafe_allow_html=True)
    _, col, _ = st.columns([1, 2, 1])
    with col:
        username = st.text_input("👤 Username", placeholder="admin / analyst / demo")
        password = st.text_input("🔑 Password", type="password", placeholder="Enter password")
        if st.button("🚀 Sign In", use_container_width=True):
            if check_login(username, password):
                st.session_state["logged_in"] = True
                st.session_state["username"]   = username
                st.rerun()
            else:
                st.error("❌ Invalid credentials. Try: admin / admin123  or  demo / demo")
        st.markdown("""<div style="text-align:center;margin-top:1rem;font-size:.78rem;color:#9d9abf">
          Demo: <b style="color:#a78bfa">admin</b>/admin123 &nbsp;|&nbsp;
          <b style="color:#22d3ee">demo</b>/demo</div>""", unsafe_allow_html=True)


if not st.session_state["logged_in"]:
    login_page()
    st.stop()


# ─── Real-Time Simulation ────────────────────────────────────────────────────────
def simulate_realtime_customers(n: int = 12) -> pd.DataFrame:
    rng = np.random.default_rng(int(time.time()) % 99999)
    contracts = ["Month-to-month", "One year", "Two year"]
    internets = ["DSL", "Fiber optic", "No"]
    payments  = ["Electronic check", "Mailed check", "Bank transfer", "Credit card"]
    yn        = ["Yes", "No"]
    rows = []
    for i in range(n):
        tenure  = int(rng.integers(1, 73))
        monthly = round(float(rng.uniform(18, 120)), 2)
        inp = dict(
            SeniorCitizen=int(rng.integers(0,2)), tenure=tenure,
            MonthlyCharges=monthly, TotalCharges=round(monthly*tenure,2),
            gender=rng.choice(["Male","Female"]), Partner=rng.choice(yn),
            Dependents=rng.choice(yn), PhoneService=rng.choice(yn),
            MultipleLines=rng.choice(["No","Yes","No phone service"]),
            InternetService=rng.choice(internets),
            OnlineSecurity=rng.choice(["No","Yes","No internet service"]),
            OnlineBackup=rng.choice(["No","Yes","No internet service"]),
            DeviceProtection=rng.choice(["No","Yes","No internet service"]),
            TechSupport=rng.choice(["No","Yes","No internet service"]),
            StreamingTV=rng.choice(["No","Yes","No internet service"]),
            StreamingMovies=rng.choice(["No","Yes","No internet service"]),
            Contract=rng.choice(contracts), PaperlessBilling=rng.choice(yn),
            PaymentMethod=rng.choice(payments),
        )
        try:
            prob = float(model.predict_proba(encode_customer(inp))[0][1])
        except Exception:
            prob = 0.5
        rows.append({
            "Customer ID": f"SIM-{1000+i}",
            "Tenure":      tenure,
            "Monthly $":   monthly,
            "Contract":    inp["Contract"][:12],
            "Internet":    inp["InternetService"],
            "Churn Prob %": round(prob*100, 1),
            "Risk": ("🔴 Critical" if prob>=.75 else "🟠 High" if prob>=.5
                     else "🟡 Medium" if prob>=.3 else "🟢 Low"),
        })
    return pd.DataFrame(rows).sort_values("Churn Prob %", ascending=False)


# ─── Customer Segmentation ───────────────────────────────────────────────────────
@st.cache_data
def run_segmentation(_df, n_clusters=4):
    feats = ["tenure","MonthlyCharges","TotalCharges"]
    sub   = _df[feats].copy()
    sub["TotalCharges"] = pd.to_numeric(sub["TotalCharges"], errors="coerce").fillna(0)
    scaled = StandardScaler().fit_transform(sub)
    labels = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit_predict(scaled)
    result = _df.copy()
    result["TotalCharges"] = pd.to_numeric(result["TotalCharges"], errors="coerce").fillna(0)
    result["Segment"] = labels
    return result

SEG_META = {
    0: {"name":"💎 Champions",     "color":"#22d3ee",
        "desc":"Long tenure, high spend. Most loyal — reward them."},
    1: {"name":"⚠️ At-Risk",       "color":"#f87171",
        "desc":"Short tenure, high charges. Likely to churn soon."},
    2: {"name":"🌱 New Customers", "color":"#a78bfa",
        "desc":"Low tenure, low spend. Need onboarding attention."},
    3: {"name":"⭐ High Value",     "color":"#fbbf24",
        "desc":"Medium tenure, very high charges. Upsell & retain."},
}

def seg_scatter_3d(seg_df):
    fig = go.Figure()
    fallback_colors = ["#22d3ee", "#f87171", "#a78bfa", "#fbbf24", "#34d399", "#fb7185"]
    for sid in sorted(seg_df["Segment"].unique()):
        meta = SEG_META.get(int(sid), {
            "name": f"Segment {int(sid) + 1}",
            "color": fallback_colors[int(sid) % len(fallback_colors)],
            "desc": "Data-driven customer cluster.",
        })
        mask = seg_df["Segment"] == sid
        if not mask.any(): continue
        fig.add_trace(go.Scatter3d(
            x=seg_df[mask]["tenure"], y=seg_df[mask]["MonthlyCharges"],
            z=seg_df[mask]["TotalCharges"], mode="markers", name=meta["name"],
            marker=dict(size=5, color=meta["color"], opacity=0.8,
                        line=dict(color="rgba(255,255,255,0.1)", width=0.5)),
            hovertemplate=f"<b>{meta['name']}</b><br>Tenure:%{{x}}mo Monthly:$%{{y:.0f}}<extra></extra>",
        ))
    fig.update_layout(
        scene=dict(
            xaxis=dict(title=dict(text="Tenure (mo)"), backgroundcolor="rgba(3,7,18,0.8)",
                       gridcolor="rgba(45,43,69,0.5)", showbackground=True, tickfont=dict(color="#9d9abf",size=9)),
            yaxis=dict(title=dict(text="Monthly $"), backgroundcolor="rgba(3,7,18,0.8)",
                       gridcolor="rgba(45,43,69,0.5)", showbackground=True, tickfont=dict(color="#9d9abf",size=9)),
            zaxis=dict(title=dict(text="Total $"), backgroundcolor="rgba(3,7,18,0.8)",
                       gridcolor="rgba(45,43,69,0.5)", showbackground=True, tickfont=dict(color="#9d9abf",size=9)),
            bgcolor="rgba(3,7,18,0.9)", camera=dict(eye=dict(x=1.5,y=1.5,z=1.0)),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(bgcolor="rgba(19,17,31,0.8)", bordercolor="rgba(124,58,237,0.3)",
                    borderwidth=1, font=dict(color="#d4d0f0",size=11)),
        margin=dict(t=10,b=10,l=0,r=0), height=460, font={"family":"Inter"},
    )
    return fig


# ─── AI Recommendation Engine ────────────────────────────────────────────────────
def ai_recommendations(prob: float, inputs: dict) -> list:
    recs = []
    tenure=inputs["tenure"]; contract=inputs["Contract"]
    internet=inputs["InternetService"]; payment=inputs["PaymentMethod"]
    monthly=inputs["MonthlyCharges"]; senior=inputs["SeniorCitizen"]
    online_sec=inputs.get("OnlineSecurity","No"); tech_sup=inputs.get("TechSupport","No")

    if contract == "Month-to-month":
        recs.append({"priority":"🔴 High","action":"Contract Upgrade Offer",
            "detail":"Offer 20% discount on annual plan. Month-to-month customers churn 3× more.",
            "impact":"↓ 35% churn risk","color":"#f87171"})
    if payment == "Electronic check":
        recs.append({"priority":"🟠 Medium","action":"Auto-Pay Incentive",
            "detail":"Offer $5/month credit for switching to bank transfer or credit card.",
            "impact":"↓ 15% churn risk","color":"#fbbf24"})
    if tenure < 6:
        recs.append({"priority":"🔴 High","action":"Onboarding Campaign",
            "detail":"Trigger 30-day onboarding sequence. First 6 months = highest churn window.",
            "impact":"↓ 25% churn risk","color":"#f87171"})
    if internet == "Fiber optic" and online_sec == "No":
        recs.append({"priority":"🟠 Medium","action":"Security Bundle Upsell",
            "detail":"Fiber users with Online Security churn 40% less. Offer free 3-month trial.",
            "impact":"↓ 20% churn risk","color":"#fbbf24"})
    if internet == "Fiber optic" and tech_sup == "No":
        recs.append({"priority":"🟡 Low","action":"Tech Support Add-On",
            "detail":"Proactively offer Tech Support — reduces frustration-driven churn.",
            "impact":"↓ 12% churn risk","color":"#fde68a"})
    if monthly > 80 and tenure < 24:
        recs.append({"priority":"🔴 High","action":"Loyalty Discount",
            "detail":f"High spend (${monthly:.0f}/mo) + low tenure = high risk. Offer 10% loyalty discount.",
            "impact":"↓ 18% churn risk","color":"#f87171"})
    if senior == 1:
        recs.append({"priority":"🟡 Low","action":"Senior Care Program",
            "detail":"Enroll in dedicated senior support with simplified billing.",
            "impact":"↓ 10% churn risk","color":"#fde68a"})
    if prob < 0.3:
        recs.append({"priority":"🟢 Opportunity","action":"Upsell Premium Services",
            "detail":"Low churn risk — ideal time to offer streaming bundles or device protection.",
            "impact":"↑ Revenue opportunity","color":"#34d399"})
    if not recs:
        recs.append({"priority":"🟢 Good","action":"Maintain Engagement",
            "detail":"Customer profile is healthy. Send quarterly satisfaction survey.",
            "impact":"Maintain loyalty","color":"#34d399"})
    return recs


# ─── Batch Prediction ────────────────────────────────────────────────────────────
def batch_predict(uploaded_df: pd.DataFrame) -> pd.DataFrame:
    results = []
    for _, row in uploaded_df.iterrows():
        try:
            tc = str(row.get("TotalCharges","0")).strip()
            inp = dict(
                SeniorCitizen=int(row.get("SeniorCitizen",0)),
                tenure=float(row.get("tenure",12)),
                MonthlyCharges=float(row.get("MonthlyCharges",65)),
                TotalCharges=float(tc if tc else 0),
                gender=str(row.get("gender","Male")),
                Partner=str(row.get("Partner","No")),
                Dependents=str(row.get("Dependents","No")),
                PhoneService=str(row.get("PhoneService","Yes")),
                MultipleLines=str(row.get("MultipleLines","No")),
                InternetService=str(row.get("InternetService","DSL")),
                OnlineSecurity=str(row.get("OnlineSecurity","No")),
                OnlineBackup=str(row.get("OnlineBackup","No")),
                DeviceProtection=str(row.get("DeviceProtection","No")),
                TechSupport=str(row.get("TechSupport","No")),
                StreamingTV=str(row.get("StreamingTV","No")),
                StreamingMovies=str(row.get("StreamingMovies","No")),
                Contract=str(row.get("Contract","Month-to-month")),
                PaperlessBilling=str(row.get("PaperlessBilling","Yes")),
                PaymentMethod=str(row.get("PaymentMethod","Electronic check")),
            )
            prob = float(model.predict_proba(encode_customer(inp))[0][1])
            results.append({
                "CustomerID":   row.get("customerID", f"ROW-{len(results)+1}"),
                "Churn_Prob_%": round(prob*100,1),
                "Prediction":   "Churn" if prob>=0.5 else "Stay",
                "Risk_Tier":    ("Critical" if prob>=.75 else "High" if prob>=.5
                                 else "Medium" if prob>=.3 else "Low"),
                "Monthly_$":    row.get("MonthlyCharges",""),
                "Tenure_mo":    row.get("tenure",""),
                "Contract":     row.get("Contract",""),
            })
        except Exception as e:
            results.append({"CustomerID":row.get("customerID","?"),
                             "Churn_Prob_%":"Error","Prediction":str(e)[:40],
                             "Risk_Tier":"","Monthly_$":"","Tenure_mo":"","Contract":""})
    return pd.DataFrame(results)


# ─── Alert System ────────────────────────────────────────────────────────────────
def get_alerts() -> list:
    alerts = []
    mtm_churn  = (df[df["Contract"]=="Month-to-month"]["Churn"]=="Yes").mean()
    fiber_churn= (df[df["InternetService"]=="Fiber optic"]["Churn"]=="Yes").mean()
    new_churn  = (df[df["tenure"]<=6]["Churn"]=="Yes").mean()
    overall    = (df["Churn"]=="Yes").mean()
    if mtm_churn > 0.4:
        alerts.append({"icon":"🔴","title":"High Month-to-Month Churn",
            "msg":f"{mtm_churn:.0%} of month-to-month customers are churning. Launch contract upgrade campaign.",
            "time":"Now","severity":"critical"})
    if fiber_churn > 0.35:
        alerts.append({"icon":"🟠","title":"Fiber Optic Segment Alert",
            "msg":f"{fiber_churn:.0%} fiber customers churning. Check service quality & pricing.",
            "time":"Today","severity":"high"})
    if new_churn > 0.4:
        alerts.append({"icon":"🟠","title":"New Customer Churn Risk",
            "msg":f"{new_churn:.0%} of customers ≤6 months tenure are churning. Strengthen onboarding.",
            "time":"Today","severity":"high"})
    if overall > 0.25:
        alerts.append({"icon":"🟡","title":"Overall Churn Rate Elevated",
            "msg":f"Overall churn at {overall:.0%}. Industry benchmark is ~15-20%.",
            "time":"This week","severity":"medium"})
    alerts.append({"icon":"💡","title":"Opportunity: Electronic Check Users",
        "msg":f"{(df['PaymentMethod']=='Electronic check').mean():.0%} customers use electronic check — highest churn payment method.",
        "time":"Ongoing","severity":"info"})
    return alerts


# ─── Why Churn Happens ───────────────────────────────────────────────────────────
WHY_CHURN = [
    {"icon":"📋","title":"Month-to-Month Contracts","color":"#f87171",
     "stat":"3× higher churn",
     "desc":"No switching cost means customers can leave anytime. Annual contracts create commitment and reduce churn dramatically.",
     "fix":"Offer discounted long-term contracts with added perks."},
    {"icon":"💸","title":"High Monthly Charges","color":"#fbbf24",
     "stat":"↑ Risk above $80/mo",
     "desc":"Customers paying high fees without perceiving equivalent value shop competitors. Price sensitivity is highest in the first 12 months.",
     "fix":"Bundle services to increase perceived value at same price point."},
    {"icon":"🌐","title":"Fiber Optic Without Add-Ons","color":"#a78bfa",
     "stat":"40% churn without security",
     "desc":"Fiber customers pay premium but skip security/support add-ons. When issues arise, they have no safety net and churn out of frustration.",
     "fix":"Proactively offer Online Security and Tech Support trials."},
    {"icon":"💳","title":"Electronic Check Payment","color":"#f87171",
     "stat":"Highest churn payment method",
     "desc":"Electronic check users show the highest churn rates, possibly indicating lower engagement or financial instability.",
     "fix":"Incentivize auto-pay with monthly bill credits."},
    {"icon":"🚀","title":"Low Tenure (First 12 Months)","color":"#fbbf24",
     "stat":"Highest churn window",
     "desc":"The first year is the most critical retention period. Customers who haven't experienced full value are most likely to leave.",
     "fix":"Implement structured 90-day onboarding with check-in calls."},
    {"icon":"👴","title":"Senior Citizens","color":"#22d3ee",
     "stat":"Higher churn vs non-seniors",
     "desc":"Senior customers often struggle with complex billing and technical issues. Without dedicated support, frustration leads to churn.",
     "fix":"Create a Senior Care program with a dedicated support line."},
    {"icon":"📄","title":"Paperless Billing","color":"#a78bfa",
     "stat":"Slightly higher churn",
     "desc":"Paperless billing customers are more digitally engaged but also more price-aware and comparison-shop more actively online.",
     "fix":"Send personalized digital retention offers to paperless customers."},
    {"icon":"👥","title":"No Partner / No Dependents","color":"#34d399",
     "stat":"More likely to churn",
     "desc":"Single customers without dependents have more flexibility to switch providers. Family plans create switching costs.",
     "fix":"Promote family/household bundle plans with shared benefits."},
]

def why_churn_chart():
    factors = {
        "Month-to-month": (df["Contract"]=="Month-to-month"),
        "Fiber Optic":    (df["InternetService"]=="Fiber optic"),
        "Elec. Check":    (df["PaymentMethod"]=="Electronic check"),
        "Tenure ≤6mo":    (df["tenure"]<=6),
        "Senior":         (df["SeniorCitizen"]==1),
        "No Partner":     (df["Partner"]=="No"),
        "Paperless":      (df["PaperlessBilling"]=="Yes"),
        "Two Year":       (df["Contract"]=="Two year"),
    }
    names = list(factors.keys())
    rates = [(df[m]["Churn"]=="Yes").mean()*100 for m in factors.values()]
    colors= ["#f87171" if r>40 else "#fbbf24" if r>25 else "#34d399" for r in rates]
    fig = go.Figure(go.Bar(
        x=names, y=rates, marker=dict(color=colors, line=dict(color="rgba(0,0,0,0)")),
        text=[f"{r:.0f}%" for r in rates], textposition="outside",
        textfont=dict(color="#d4d0f0",size=11),
        hovertemplate="%{x}<br>Churn Rate: %{y:.1f}%<extra></extra>",
    ))
    overall = (df["Churn"]=="Yes").mean()*100
    fig.add_hline(y=overall, line_dash="dash", line_color="#a78bfa", line_width=1.5,
                  annotation_text=f"Overall: {overall:.0f}%",
                  annotation_font_color="#a78bfa", annotation_font_size=10)
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={"color":"#9d9abf","family":"Inter"},
        xaxis=dict(gridcolor="rgba(45,43,69,0.3)"),
        yaxis=dict(gridcolor="rgba(45,43,69,0.4)", title="Churn Rate %",
                   range=[0, max(rates)+15]),
        margin=dict(t=20,b=40,l=40,r=20), height=320,
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#                               SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:1rem 0 .5rem">
      <div style="font-size:2.5rem">🛡️</div>
      <div style="font-family:'Space Grotesk',sans-serif;font-size:1.2rem;font-weight:700;
                  background:linear-gradient(135deg,#a78bfa,#22d3ee);
                  -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                  background-clip:text;">ChurnShield AI</div>
      <div style="font-size:.72rem;color:#9d9abf;margin-top:.2rem">Customer Intelligence Platform</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div style="font-size:.78rem;color:#9d9abf;text-align:center;margin-bottom:.8rem">
      Signed in as <b style="color:#22d3ee">{st.session_state["username"]}</b>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Sign Out", use_container_width=True):
        st.session_state["logged_in"] = False
        st.session_state["username"] = ""
        st.rerun()

    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section-header">👤 Demographics</div>', unsafe_allow_html=True)
    gender     = st.selectbox("Gender",        ["Male", "Female"])
    senior     = st.checkbox("Senior Citizen", value=False)
    partner    = st.selectbox("Partner",        ["No", "Yes"])
    dependents = st.selectbox("Dependents",     ["No", "Yes"])

    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-section-header">📋 Account Details</div>', unsafe_allow_html=True)
    tenure   = st.slider("Tenure (months)", 1, 72, 12)
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment  = st.selectbox("Payment Method",
                             ["Electronic check", "Mailed check", "Bank transfer", "Credit card"])
    monthly  = st.number_input("Monthly Charges ($)", 18.0, 120.0, 65.0, step=0.5)
    total    = st.number_input("Total Charges ($)", 0.0, 9000.0,
                                float(round(monthly * tenure, 2)), step=10.0)

    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-section-header">📡 Services</div>', unsafe_allow_html=True)
    phone      = st.selectbox("Phone Service",    ["Yes", "No"])
    multi_lines = st.selectbox("Multiple Lines",  ["No", "Yes", "No phone service"])
    internet   = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

    if internet != "No":
        online_sec  = st.selectbox("Online Security",   ["No", "Yes"])
        online_bk   = st.selectbox("Online Backup",     ["No", "Yes"])
        device_prot = st.selectbox("Device Protection", ["No", "Yes"])
        tech_sup    = st.selectbox("Tech Support",       ["No", "Yes"])
        stream_tv   = st.selectbox("Streaming TV",       ["No", "Yes"])
        stream_mov  = st.selectbox("Streaming Movies",   ["No", "Yes"])
    else:
        online_sec = online_bk = device_prot = tech_sup = stream_tv = stream_mov = "No internet service"

    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
    if st.button("⚡ Predict Churn Risk", use_container_width=True):
        st.session_state["run_prediction"] = True
    predict_btn = st.session_state["run_prediction"]

    # Sidebar mini stats
    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
    churn_rate_s = (df["Churn"] == "Yes").mean()
    st.markdown(f"""
    <div style="display:flex;justify-content:space-around;padding:.5rem 0">
      <div style="text-align:center">
        <div style="font-size:1.1rem;font-weight:700;color:#22d3ee;font-family:'Space Grotesk',sans-serif">{len(df)}</div>
        <div style="font-size:.65rem;color:#9d9abf;text-transform:uppercase;letter-spacing:.08em">Records</div>
      </div>
      <div style="text-align:center">
        <div style="font-size:1.1rem;font-weight:700;color:#f87171;font-family:'Space Grotesk',sans-serif">{churn_rate_s:.0%}</div>
        <div style="font-size:.65rem;color:#9d9abf;text-transform:uppercase;letter-spacing:.08em">Churn Rate</div>
      </div>
      <div style="text-align:center">
        <div style="font-size:1.1rem;font-weight:700;color:#34d399;font-family:'Space Grotesk',sans-serif">{len(FEATURE_COLS)}</div>
        <div style="font-size:.65rem;color:#9d9abf;text-transform:uppercase;letter-spacing:.08em">Features</div>
      </div>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#                               MAIN AREA
# ══════════════════════════════════════════════════════════════════════════════

# ── KPI data ──
churn_rate  = (df["Churn"] == "Yes").mean()
avg_tenure  = df["tenure"].mean()
avg_monthly = df["MonthlyCharges"].mean()
high_risk   = (df["Contract"] == "Month-to-month").mean()

# ── Hero ──
st.markdown(f"""
<div class="hero-wrap">
  <div class="hero-badge">🛡️ &nbsp; AI-Powered Churn Intelligence</div>
  <div class="hero-title">ChurnShield AI</div>
  <div class="hero-sub">
    Real-time customer churn prediction powered by Logistic Regression.
    Explore 3D risk landscapes, feature impacts, and actionable retention strategies.
  </div>
  <div class="hero-stats">
    <div class="hero-stat">
      <div class="hero-stat-val" style="color:#22d3ee">{len(df)}</div>
      <div class="hero-stat-lbl">Customer Records</div>
    </div>
    <div class="hero-stat">
      <div class="hero-stat-val">{len(FEATURE_COLS)}</div>
      <div class="hero-stat-lbl">Model Features</div>
    </div>
    <div class="hero-stat">
      <div class="hero-stat-val">{churn_rate:.0%}</div>
      <div class="hero-stat-lbl">Churn Rate</div>
    </div>
    <div class="hero-stat">
      <div class="hero-stat-val">LR</div>
      <div class="hero-stat-lbl">Algorithm</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── KPI Cards ──
k1, k2, k3, k4 = st.columns(4)
with k1:
    st.markdown(f"""<div class="glass-card">
        <div class="kpi-icon">📉</div>
        <div class="kpi-value">{churn_rate:.0%}</div>
        <div class="kpi-label">Overall Churn Rate</div>
        <div class="kpi-delta" style="color:#f87171">▲ Needs attention</div>
    </div>""", unsafe_allow_html=True)
with k2:
    st.markdown(f"""<div class="glass-card">
        <div class="kpi-icon">⏱️</div>
        <div class="kpi-value">{avg_tenure:.0f}mo</div>
        <div class="kpi-label">Avg. Customer Tenure</div>
        <div class="kpi-delta" style="color:#9d9abf">Across all contracts</div>
    </div>""", unsafe_allow_html=True)
with k3:
    st.markdown(f"""<div class="glass-card">
        <div class="kpi-icon">💰</div>
        <div class="kpi-value">${avg_monthly:.0f}</div>
        <div class="kpi-label">Avg. Monthly Charge</div>
        <div class="kpi-delta" style="color:#9d9abf">Per customer</div>
    </div>""", unsafe_allow_html=True)
with k4:
    st.markdown(f"""<div class="glass-card">
        <div class="kpi-icon">⚠️</div>
        <div class="kpi-value">{high_risk:.0%}</div>
        <div class="kpi-label">Month-to-Month</div>
        <div class="kpi-delta" style="color:#fbbf24">▲ Highest churn risk</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Tabs ──
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "⚡ Prediction",
    "🤖 AI Engine",
    "🌐 3D Analytics",
    "👥 Segments",
    "📊 Dashboard",
    "📁 Batch CSV",
    "❓ Why Churn",
])


# ══════════════════════════════════ TAB 1 ════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-header"><span class="section-icon">⚡</span><span class="section-title">Customer Churn Prediction</span></div>', unsafe_allow_html=True)

    if predict_btn:
        inputs = dict(
            SeniorCitizen=int(senior), tenure=tenure,
            MonthlyCharges=monthly, TotalCharges=total,
            gender=gender, Partner=partner, Dependents=dependents,
            PhoneService=phone, MultipleLines=multi_lines,
            InternetService=internet, OnlineSecurity=online_sec,
            OnlineBackup=online_bk, DeviceProtection=device_prot,
            TechSupport=tech_sup, StreamingTV=stream_tv,
            StreamingMovies=stream_mov, Contract=contract,
            PaperlessBilling=paperless, PaymentMethod=payment,
        )
        X       = encode_customer(inputs)
        prob    = float(model.predict_proba(X)[0][1])
        pred    = int(prob >= 0.5)
        coef    = model.coef_[0]
        vals    = X.values[0]
        contrib = coef * vals
        shap_vals = np.zeros(len(FEATURE_COLS))  # default, updated after SHAP section

        pos_idx      = np.argsort(contrib)[-5:][::-1]
        neg_idx      = np.argsort(contrib)[:5]
        risk_factors = [FEATURE_COLS[i].replace("_", " ") for i in pos_idx if contrib[i] > 0.001]
        prot_factors = [FEATURE_COLS[i].replace("_", " ") for i in neg_idx if contrib[i] < -0.001]

        # ── Row 1: Gauge | Result | Factors ──
        col_g, col_r, col_f = st.columns([1.1, 1.2, 1.7])

        with col_g:
            st.markdown('<div class="chart-3d-wrap">', unsafe_allow_html=True)
            st.plotly_chart(gauge_3d(prob), use_container_width=True, config={"displayModeBar": False}, key="prediction_gauge")
            st.markdown('</div>', unsafe_allow_html=True)

        with col_r:
            if pred == 1:
                st.markdown(f"""<div class="result-card result-risk">
                    <div class="result-label" style="color:#f87171">⚠️ Churn Risk Detected</div>
                    <div class="result-prob" style="color:#f87171">{prob:.0%}</div>
                    <div class="result-desc">High likelihood of churning. Immediate retention action recommended.</div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""<div class="result-card result-safe">
                    <div class="result-label" style="color:#34d399">✅ Low Churn Risk</div>
                    <div class="result-prob" style="color:#34d399">{prob:.0%}</div>
                    <div class="result-desc">Customer is likely to stay. Consider upsell or loyalty rewards.</div>
                </div>""", unsafe_allow_html=True)

            # Tier badge
            if prob >= .75:
                tier_cls, tier_txt = "tier-critical", "🔴 Critical Risk"
            elif prob >= .5:
                tier_cls, tier_txt = "tier-high", "🟠 High Risk"
            elif prob >= .3:
                tier_cls, tier_txt = "tier-medium", "🟡 Medium Risk"
            else:
                tier_cls, tier_txt = "tier-low", "🟢 Low Risk"

            st.markdown(f"""<div style="text-align:center;margin-top:.8rem">
                <span class="tier-badge {tier_cls}">{tier_txt}</span>
            </div>""", unsafe_allow_html=True)

            # Customer summary
            st.markdown(f"""
            <div style="background:rgba(19,17,31,0.5);border:1px solid rgba(34,211,238,0.15);
                        border-radius:12px;padding:1rem;margin-top:.8rem;font-size:.82rem;color:#9d9abf">
              <div style="display:grid;grid-template-columns:1fr 1fr;gap:.4rem">
                <div>📅 Tenure: <b style="color:#d4d0f0">{tenure} mo</b></div>
                <div>💳 Contract: <b style="color:#d4d0f0">{contract[:10]}</b></div>
                <div>💰 Monthly: <b style="color:#d4d0f0">${monthly:.0f}</b></div>
                <div>🌐 Internet: <b style="color:#d4d0f0">{internet}</b></div>
              </div>
            </div>
            """, unsafe_allow_html=True)

        with col_f:
            st.markdown('<div class="section-header"><span class="section-icon">🎯</span><span class="section-title">Key Factors</span></div>', unsafe_allow_html=True)
            if risk_factors:
                st.markdown("<small style='color:#f87171;font-weight:600'>🔴 Churn-driving factors</small>", unsafe_allow_html=True)
                pills = "".join([f'<span class="pill-risk">⬆ {f}</span>' for f in risk_factors])
                st.markdown(pills, unsafe_allow_html=True)
            if prot_factors:
                st.markdown("<small style='color:#34d399;font-weight:600;margin-top:.5rem;display:block'>🟢 Retention-supporting factors</small>", unsafe_allow_html=True)
                pills = "".join([f'<span class="pill-safe">⬇ {f}</span>' for f in prot_factors])
                st.markdown(pills, unsafe_allow_html=True)

            # Advice
            advice = []
            if contract == "Month-to-month":
                advice.append(("📋", "Offer a <b style='color:#22d3ee'>discounted annual contract</b> to lock in loyalty."))
            if internet == "Fiber optic" and online_sec == "No":
                advice.append(("🔒", "Recommend <b style='color:#a78bfa'>Online Security add-on</b> — fiber users who add security churn less."))
            if payment == "Electronic check":
                advice.append(("💳", "Incentivize switching to <b style='color:#a78bfa'>auto-pay</b> (bank transfer/credit card)."))
            if tenure < 12:
                advice.append(("🚀", "Customer is in the <b style='color:#a78bfa'>high-churn first-year window</b> — trigger an onboarding check-in."))
            if not advice:
                advice.append(("🎁", "Profile looks healthy. Consider a <b style='color:#22d3ee'>loyalty reward</b> to maintain satisfaction."))

            items_html = "".join([
                f'<div class="advice-item"><span class="advice-dot">{icon}</span><span>{text}</span></div>'
                for icon, text in advice
            ])
            st.markdown(f"""
            <div class="advice-glass">
              <div class="advice-title">💡 Recommended Actions</div>
              {items_html}
            </div>
            """, unsafe_allow_html=True)

        # ── Row 2: 3D Feature Impact ──
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-header"><span class="section-icon">🌐</span><span class="section-title">3D Feature Impact Analysis</span></div>', unsafe_allow_html=True)
        st.markdown("<small style='color:#9d9abf'>Drag to rotate · Scroll to zoom · Hover for details</small>", unsafe_allow_html=True)
        st.markdown('<div class="chart-3d-wrap">', unsafe_allow_html=True)
        st.plotly_chart(feature_impact_3d(X), use_container_width=True, config={"displayModeBar": True}, key="prediction_feature_impact")
        st.markdown('</div>', unsafe_allow_html=True)

        # ── Row 3: SHAP | Voice | PDF ──
        st.markdown("<br>", unsafe_allow_html=True)
        shap_col, action_col = st.columns([2, 1])

        with shap_col:
            st.markdown("""
            <div class="section-header">
              <span class="section-icon">🤖</span>
              <span class="section-title">SHAP Explainability</span>
              <span style="font-size:.75rem;color:#9d9abf;margin-left:.5rem">
                — Why did the model predict this?
              </span>
            </div>
            """, unsafe_allow_html=True)
            with st.spinner("Computing SHAP values..."):
                X_background = df.copy()
                # encode all rows for background
                bg_rows = []
                for _, row in X_background.iterrows():
                    try:
                        bg_rows.append(encode_customer({
                            "SeniorCitizen": int(row["SeniorCitizen"]),
                            "tenure": row["tenure"],
                            "MonthlyCharges": row["MonthlyCharges"],
                            "TotalCharges": row["TotalCharges"] if pd.notna(row["TotalCharges"]) else 0,
                            "gender": row["gender"], "Partner": row["Partner"],
                            "Dependents": row["Dependents"], "PhoneService": row["PhoneService"],
                            "MultipleLines": row["MultipleLines"],
                            "InternetService": row["InternetService"],
                            "OnlineSecurity": row["OnlineSecurity"],
                            "OnlineBackup": row["OnlineBackup"],
                            "DeviceProtection": row["DeviceProtection"],
                            "TechSupport": row["TechSupport"],
                            "StreamingTV": row["StreamingTV"],
                            "StreamingMovies": row["StreamingMovies"],
                            "Contract": row["Contract"],
                            "PaperlessBilling": row["PaperlessBilling"],
                            "PaymentMethod": row["PaymentMethod"],
                        }).values[0])
                    except Exception:
                        pass
                X_bg = pd.DataFrame(bg_rows, columns=FEATURE_COLS)
                fig_shap, shap_vals = shap_chart(X, X_bg)
            st.markdown("""
            <div style="background:rgba(34,211,238,0.05);border:1px solid rgba(34,211,238,0.2);
                        border-radius:10px;padding:.6rem 1rem;margin-bottom:.6rem;font-size:.82rem;color:#9d9abf">
              🔴 <b style="color:#f87171">Red bars</b> push prediction toward churn &nbsp;|&nbsp;
              🔵 <b style="color:#22d3ee">Blue bars</b> push prediction away from churn &nbsp;|&nbsp;
              <b style="color:#a78bfa">Dashed line</b> = base (average) prediction
            </div>
            """, unsafe_allow_html=True)
            st.markdown('<div class="chart-3d-wrap">', unsafe_allow_html=True)
            st.plotly_chart(fig_shap, use_container_width=True, config={"displayModeBar": False}, key="prediction_shap")
            st.markdown('</div>', unsafe_allow_html=True)

        with action_col:
            # ── Voice Assistant ──
            st.markdown("""
            <div class="section-header">
              <span class="section-icon">🔊</span>
              <span class="section-title">AI Voice Assistant</span>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("""
            <div style="background:rgba(167,139,250,0.07);border:1px solid rgba(167,139,250,0.25);
                        border-radius:12px;padding:1rem;margin-bottom:1rem;font-size:.85rem;color:#9d9abf;line-height:1.6">
              Click below to hear an AI-generated voice summary of this prediction,
              including risk tier, key factors, and recommended actions.
            </div>
            """, unsafe_allow_html=True)
            if st.button("🔊 Generate Voice Summary", use_container_width=True):
                with st.spinner("Generating audio..."):
                    audio_bytes = generate_voice_summary(
                        prob, pred, risk_factors, prot_factors,
                        advice, tenure, contract, monthly
                    )
                st.audio(audio_bytes, format="audio/mp3")
                st.success("✅ Audio ready — press play above!")

            st.markdown("<br>", unsafe_allow_html=True)

            # ── PDF Report ──
            st.markdown("""
            <div class="section-header">
              <span class="section-icon">📄</span>
              <span class="section-title">PDF Report</span>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("""
            <div style="background:rgba(34,211,238,0.07);border:1px solid rgba(34,211,238,0.25);
                        border-radius:12px;padding:1rem;margin-bottom:1rem;font-size:.85rem;color:#9d9abf;line-height:1.6">
              Download a full PDF report with customer profile, risk analysis,
              SHAP feature importance, and recommended actions.
            </div>
            """, unsafe_allow_html=True)
            with st.spinner("Building PDF..."):
                pdf_bytes = generate_pdf_report(
                    inputs, prob, pred,
                    risk_factors, prot_factors,
                    advice, shap_vals
                )
            st.download_button(
                label="📥 Download PDF Report",
                data=pdf_bytes,
                file_name=f"churnshield_report_{inputs.get('tenure','')}_mo.pdf",
                mime="application/pdf",
                use_container_width=True,
            )

    else:
        st.markdown("""
        <div class="placeholder-wrap">
          <div class="placeholder-icon">⚡</div>
          <div class="placeholder-title">Configure Customer Profile</div>
          <div class="placeholder-sub">
            Use the sidebar to set customer attributes, then click
            <b style="color:#a78bfa">Predict Churn Risk</b> to see real-time predictions,
            3D feature impact analysis, risk factors, and actionable recommendations.
          </div>
          <div class="feature-grid">
            <div class="feature-item"><div class="feature-item-icon">🎯</div><div class="feature-item-lbl">3D Gauge</div></div>
            <div class="feature-item"><div class="feature-item-icon">🌐</div><div class="feature-item-lbl">3D Impact</div></div>
            <div class="feature-item"><div class="feature-item-icon">🤖</div><div class="feature-item-lbl">SHAP AI</div></div>
            <div class="feature-item"><div class="feature-item-icon">🔊</div><div class="feature-item-lbl">Voice AI</div></div>
            <div class="feature-item"><div class="feature-item-icon">📄</div><div class="feature-item-lbl">PDF Report</div></div>
          </div>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════ TAB 2 — AI ENGINE ═══════════════════════════
with tab2:
    st.markdown('<div class="section-header"><span class="section-icon">🤖</span><span class="section-title">AI Recommendation Engine, Live Simulation & Alerts</span></div>', unsafe_allow_html=True)

    current_inputs = dict(
        SeniorCitizen=int(senior), tenure=tenure,
        MonthlyCharges=monthly, TotalCharges=total,
        gender=gender, Partner=partner, Dependents=dependents,
        PhoneService=phone, MultipleLines=multi_lines,
        InternetService=internet, OnlineSecurity=online_sec,
        OnlineBackup=online_bk, DeviceProtection=device_prot,
        TechSupport=tech_sup, StreamingTV=stream_tv,
        StreamingMovies=stream_mov, Contract=contract,
        PaperlessBilling=paperless, PaymentMethod=payment,
    )
    current_prob = float(model.predict_proba(encode_customer(current_inputs))[0][1])
    recs = ai_recommendations(current_prob, current_inputs)

    ai_col, live_col = st.columns([1.15, 1])
    with ai_col:
        st.markdown("""
        <div style="background:rgba(124,58,237,0.06);border:1px solid rgba(124,58,237,0.2);
                    border-radius:12px;padding:.8rem 1.2rem;margin-bottom:.8rem">
          <b style="color:#22d3ee">Next Best Actions</b>
          <span style="color:#9d9abf;font-size:.85rem;margin-left:.8rem">Personalized from the sidebar profile</span>
        </div>
        """, unsafe_allow_html=True)
        for rec in recs:
            st.markdown(f"""
            <div class="why-card" style="border-left-color:{rec['color']}">
              <div class="why-title" style="color:{rec['color']}">{rec['priority']} · {rec['action']}</div>
              <div class="why-desc">{rec['detail']}</div>
              <div style="font-size:.78rem;color:#d4d0f0;margin-top:.55rem">Expected impact: <b style="color:{rec['color']}">{rec['impact']}</b></div>
            </div>
            """, unsafe_allow_html=True)

    with live_col:
        st.markdown("""
        <div style="background:rgba(52,211,153,0.06);border:1px solid rgba(52,211,153,0.2);
                    border-radius:12px;padding:.8rem 1.2rem;margin-bottom:.8rem">
          <b style="color:#34d399"><span class="live-dot"></span>Real-Time Data Simulation</b>
          <span style="color:#9d9abf;font-size:.82rem;margin-left:.5rem">New synthetic customers each refresh</span>
        </div>
        """, unsafe_allow_html=True)
        sim_count = st.slider("Customers in live feed", 5, 30, 12, key="sim_count")
        if st.button("Refresh Live Feed", use_container_width=True):
            st.session_state["last_sim_refresh"] = datetime.now().strftime("%H:%M:%S")
        sim_df = simulate_realtime_customers(sim_count)
        st.dataframe(
            sim_df,
            use_container_width=True,
            height=360,
            column_config={
                "Churn Prob %": st.column_config.ProgressColumn(
                    "Churn Prob %",
                    min_value=0,
                    max_value=100,
                    format="%.1f%%",
                )
            },
        )
        st.caption(f"Last refresh: {st.session_state.get('last_sim_refresh', datetime.now().strftime('%H:%M:%S'))}")

    st.markdown("<br>", unsafe_allow_html=True)
    alert_col, email_col = st.columns([1, 1])
    with alert_col:
        st.markdown('<div class="section-header"><span class="section-icon">🚨</span><span class="section-title">Alert System</span></div>', unsafe_allow_html=True)
        for alert in get_alerts():
            st.markdown(f"""
            <div class="alert-item">
              <span class="alert-icon">{alert['icon']}</span>
              <div>
                <div class="alert-body"><b>{alert['title']}</b><br>{alert['msg']}</div>
                <div class="alert-time">{alert['time']} · {alert['severity'].title()}</div>
              </div>
            </div>
            """, unsafe_allow_html=True)

    with email_col:
        st.markdown('<div class="section-header"><span class="section-icon">✉️</span><span class="section-title">Email / Alert Draft</span></div>', unsafe_allow_html=True)
        recipient = st.text_input("Recipient", "retention-team@company.com")
        subject = st.text_input("Subject", f"Churn Alert: {current_prob:.0%} risk customer")
        email_body = (
            f"Hi team,\n\n"
            f"A customer profile is currently scoring {current_prob:.1%} churn risk.\n\n"
            f"Recommended action: {recs[0]['action']}\n"
            f"Why: {recs[0]['detail']}\n"
            f"Expected impact: {recs[0]['impact']}\n\n"
            f"Customer snapshot:\n"
            f"- Tenure: {tenure} months\n"
            f"- Contract: {contract}\n"
            f"- Monthly charges: ${monthly:.2f}\n"
            f"- Internet service: {internet}\n\n"
            f"Please review and trigger the retention workflow."
        )
        st.text_area("Message", email_body, height=260)
        if st.button("Simulate Sending Alert", use_container_width=True):
            st.success(f"Alert queued for {recipient}. SMTP is simulated in this demo app.")


# ══════════════════════════════════ TAB 3 — 3D Analytics ════════════════════════
with tab3:
    st.markdown('<div class="section-header"><span class="section-icon">🌐</span><span class="section-title">3D Analytics Dashboard</span></div>', unsafe_allow_html=True)

    # ── 3D Scatter ──
    st.markdown("""
    <div style="background:rgba(124,58,237,0.06);border:1px solid rgba(124,58,237,0.2);
                border-radius:12px;padding:.8rem 1.2rem;margin-bottom:.8rem">
      <b style="color:#22d3ee">🌐 3D Customer Scatter</b>
      <span style="color:#9d9abf;font-size:.85rem;margin-left:.8rem">Tenure × Monthly Charges × Total Charges — colored by churn status</span>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<div class="chart-3d-wrap">', unsafe_allow_html=True)
    st.plotly_chart(scatter_3d_churn(), use_container_width=True, key="analytics_scatter_3d")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_surf, col_bubble = st.columns(2)

    with col_surf:
        st.markdown("""
        <div style="background:rgba(124,58,237,0.06);border:1px solid rgba(124,58,237,0.2);
                    border-radius:12px;padding:.8rem 1.2rem;margin-bottom:.8rem">
          <b style="color:#22d3ee">🏔️ 3D Risk Landscape</b>
          <span style="color:#9d9abf;font-size:.82rem;margin-left:.6rem">Churn risk surface by tenure & monthly charges</span>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('<div class="chart-3d-wrap">', unsafe_allow_html=True)
        st.plotly_chart(risk_surface_3d(), use_container_width=True, key="analytics_risk_surface")
        st.markdown('</div>', unsafe_allow_html=True)

    with col_bubble:
        st.markdown("""
        <div style="background:rgba(124,58,237,0.06);border:1px solid rgba(124,58,237,0.2);
                    border-radius:12px;padding:.8rem 1.2rem;margin-bottom:.8rem">
          <b style="color:#22d3ee">🫧 3D Churn Bubble Map</b>
          <span style="color:#9d9abf;font-size:.82rem;margin-left:.6rem">Contract × Internet × Churn Rate (bubble size = risk)</span>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('<div class="chart-3d-wrap">', unsafe_allow_html=True)
        st.plotly_chart(bar_3d_churn(), use_container_width=True, key="analytics_churn_bubble")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── 2D supporting charts ──
    st.markdown('<div class="section-header"><span class="section-icon">📊</span><span class="section-title">Distribution Analysis</span></div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Overall Churn Split**")
        st.plotly_chart(churn_donut(), use_container_width=True, config={"displayModeBar": False}, key="analytics_churn_donut")
    with c2:
        st.markdown("**Tenure Distribution by Churn**")
        st.plotly_chart(tenure_hist(), use_container_width=True, config={"displayModeBar": False}, key="analytics_tenure_hist")

    c3, c4 = st.columns(2)
    with c3:
        st.markdown("**Churn by Contract Type**")
        st.plotly_chart(churn_by_col("Contract", "Contract"), use_container_width=True, config={"displayModeBar": False}, key="analytics_contract_churn")
    with c4:
        st.markdown("**Monthly Charges Distribution**")
        st.plotly_chart(monthly_violin(), use_container_width=True, config={"displayModeBar": False}, key="analytics_monthly_violin")

    c5, c6 = st.columns(2)
    with c5:
        st.markdown("**Churn by Internet Service**")
        st.plotly_chart(churn_by_col("InternetService", "Internet Service"), use_container_width=True, config={"displayModeBar": False}, key="analytics_internet_churn")
    with c6:
        st.markdown("**Churn by Payment Method**")
        st.plotly_chart(churn_by_col("PaymentMethod", "Payment Method"), use_container_width=True, config={"displayModeBar": False}, key="analytics_payment_churn")

    # Raw data
    with st.expander("🔎 Explore Raw Dataset"):
        filter_churn = st.radio("Filter by Churn", ["All", "Yes", "No"], horizontal=True)
        show_df = df if filter_churn == "All" else df[df["Churn"] == filter_churn]
        st.dataframe(show_df.reset_index(drop=True), use_container_width=True, height=300)
        st.caption(f"Showing {len(show_df):,} of {len(df):,} records")


# ══════════════════════════════════ TAB 4 — CUSTOMER SEGMENTATION ══════════════
with tab4:
    st.markdown('<div class="section-header"><span class="section-icon">👥</span><span class="section-title">Customer Segmentation</span></div>', unsafe_allow_html=True)

    cluster_count = st.slider("Number of customer segments", 3, 6, 4)
    seg_df = run_segmentation(df, cluster_count)
    seg_summary = (
        seg_df.groupby("Segment")
        .agg(
            Customers=("customerID", "count"),
            Churn_Rate=("Churn", lambda s: (s == "Yes").mean()),
            Avg_Tenure=("tenure", "mean"),
            Avg_Monthly=("MonthlyCharges", "mean"),
            Avg_Total=("TotalCharges", "mean"),
        )
        .reset_index()
    )

    seg_cols = st.columns(min(cluster_count, 4))
    for idx, row in seg_summary.iterrows():
        meta = SEG_META.get(int(row["Segment"]), {
            "name": f"Segment {int(row['Segment']) + 1}",
            "color": "#a78bfa",
            "desc": "Data-driven customer cluster.",
        })
        with seg_cols[idx % len(seg_cols)]:
            st.markdown(f"""
            <div class="seg-card" style="border-color:{meta['color']};background:rgba(19,17,31,0.55)">
              <div class="seg-title" style="color:{meta['color']}">{meta['name']}</div>
              <div class="seg-desc">{meta['desc']}</div>
              <div style="display:grid;grid-template-columns:1fr 1fr;gap:.4rem;margin-top:.8rem;font-size:.78rem;color:#9d9abf">
                <div>Customers<br><b style="color:#d4d0f0">{int(row['Customers'])}</b></div>
                <div>Churn<br><b style="color:#f87171">{row['Churn_Rate']:.0%}</b></div>
                <div>Tenure<br><b style="color:#d4d0f0">{row['Avg_Tenure']:.0f} mo</b></div>
                <div>Monthly<br><b style="color:#d4d0f0">${row['Avg_Monthly']:.0f}</b></div>
              </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    left, right = st.columns([1.3, 1])
    with left:
        st.markdown('<div class="chart-3d-wrap">', unsafe_allow_html=True)
        st.plotly_chart(seg_scatter_3d(seg_df), use_container_width=True, key="segments_scatter_3d")
        st.markdown('</div>', unsafe_allow_html=True)
    with right:
        fig_seg = px.bar(
            seg_summary,
            x="Segment",
            y="Churn_Rate",
            color="Churn_Rate",
            color_continuous_scale=["#22d3ee", "#fbbf24", "#f87171"],
            text=seg_summary["Churn_Rate"].map(lambda x: f"{x:.0%}"),
        )
        fig_seg.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font={"color":"#9d9abf","family":"Inter"},
            yaxis=dict(tickformat=".0%", gridcolor="rgba(45,43,69,0.5)", title="Churn Rate"),
            xaxis=dict(gridcolor="rgba(45,43,69,0.2)", title="Segment"),
            coloraxis_showscale=False,
            margin=dict(t=20,b=40,l=40,r=20),
            height=360,
        )
        st.plotly_chart(fig_seg, use_container_width=True, config={"displayModeBar": False}, key="segments_churn_bar")
        st.dataframe(
            seg_summary.assign(Churn_Rate=seg_summary["Churn_Rate"].map(lambda x: f"{x:.1%}")),
            use_container_width=True,
            height=220,
        )


# ══════════════════════════════════ TAB 5 — ADVANCED DASHBOARD ═════════════════
with tab5:
    st.markdown('<div class="section-header"><span class="section-icon">📊</span><span class="section-title">Advanced Dashboard</span></div>', unsafe_allow_html=True)

    dash_a, dash_b, dash_c = st.columns(3)
    with dash_a:
        st.plotly_chart(churn_donut(), use_container_width=True, config={"displayModeBar": False}, key="dashboard_churn_donut")
    with dash_b:
        contract_rates = df.groupby("Contract")["Churn"].apply(lambda s: (s == "Yes").mean()).reset_index(name="Churn Rate")
        fig_contract = px.bar(contract_rates, x="Contract", y="Churn Rate", color="Churn Rate",
                              color_continuous_scale=["#22d3ee", "#fbbf24", "#f87171"],
                              text=contract_rates["Churn Rate"].map(lambda x: f"{x:.0%}"))
        fig_contract.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font={"color":"#9d9abf","family":"Inter"},
            yaxis=dict(tickformat=".0%", gridcolor="rgba(45,43,69,0.5)"),
            coloraxis_showscale=False, margin=dict(t=20,b=30,l=35,r=10), height=260)
        st.plotly_chart(fig_contract, use_container_width=True, config={"displayModeBar": False}, key="dashboard_contract_rates")
    with dash_c:
        st.markdown('<div class="advice-glass"><div class="advice-title">🚨 Priority Alerts</div>', unsafe_allow_html=True)
        for alert in get_alerts()[:4]:
            st.markdown(f'<div class="advice-item"><span class="advice-dot">{alert["icon"]}</span><span>{alert["title"]}<br><small style="color:#9d9abf">{alert["msg"]}</small></span></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    heat = df.pivot_table(index="Contract", columns="InternetService", values="Churn", aggfunc=lambda s: (s == "Yes").mean()).fillna(0)
    fig_heat = go.Figure(go.Heatmap(
        z=heat.values,
        x=heat.columns,
        y=heat.index,
        colorscale=[[0, "#22d3ee"], [.5, "#fbbf24"], [1, "#f87171"]],
        text=[[f"{v:.0%}" for v in row] for row in heat.values],
        texttemplate="%{text}",
        hovertemplate="Contract: %{y}<br>Internet: %{x}<br>Churn: %{z:.1%}<extra></extra>",
    ))
    fig_heat.update_layout(
        title=dict(text="Churn Heatmap: Contract × Internet Service", font=dict(color="#d4d0f0", size=14)),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={"color":"#9d9abf","family":"Inter"},
        margin=dict(t=55,b=35,l=80,r=20), height=360,
    )
    st.plotly_chart(fig_heat, use_container_width=True, config={"displayModeBar": False}, key="dashboard_heatmap")

    st.markdown('<div class="section-header"><span class="section-icon">🔬</span><span class="section-title">Model Architecture & Coefficients</span></div>', unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown("""<div class="glass-card">
            <div class="kpi-icon">🧠</div>
            <div class="kpi-value" style="font-size:1.2rem">Logistic Regression</div>
            <div class="kpi-label">Algorithm</div>
            <div class="kpi-delta" style="color:#22d3ee">L2 Regularisation</div>
        </div>""", unsafe_allow_html=True)
    with m2:
        st.markdown(f"""<div class="glass-card">
            <div class="kpi-icon">🔢</div>
            <div class="kpi-value">{len(FEATURE_COLS)}</div>
            <div class="kpi-label">Input Features</div>
            <div class="kpi-delta" style="color:#9d9abf">After one-hot encoding</div>
        </div>""", unsafe_allow_html=True)
    with m3:
        st.markdown("""<div class="glass-card">
            <div class="kpi-icon">🎯</div>
            <div class="kpi-value">Binary</div>
            <div class="kpi-label">Output Classes</div>
            <div class="kpi-delta" style="color:#9d9abf">Churn / No Churn</div>
        </div>""", unsafe_allow_html=True)
    with m4:
        st.markdown("""<div class="glass-card">
            <div class="kpi-icon">⚙️</div>
            <div class="kpi-value">Sigmoid</div>
            <div class="kpi-label">Activation</div>
            <div class="kpi-delta" style="color:#9d9abf">Probability output</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── 3D Coefficient Visualization ──
    coef_df = pd.DataFrame({"feature": FEATURE_COLS, "coef": model.coef_[0]})
    coef_df = coef_df.reindex(coef_df["coef"].abs().sort_values(ascending=False).index)
    top20   = coef_df.head(20).reset_index(drop=True)

    st.markdown("""
    <div style="background:rgba(124,58,237,0.06);border:1px solid rgba(124,58,237,0.2);
                border-radius:12px;padding:.8rem 1.2rem;margin-bottom:.8rem">
      <b style="color:#22d3ee">🌐 3D Coefficient Landscape</b>
      <span style="color:#9d9abf;font-size:.85rem;margin-left:.8rem">Top 20 features — height & color = coefficient magnitude</span>
    </div>
    """, unsafe_allow_html=True)

    fig_3d_coef = go.Figure()
    for i, row in top20.iterrows():
        color = "#f87171" if row["coef"] > 0 else "#34d399"
        fig_3d_coef.add_trace(go.Scatter3d(
            x=[i, i], y=[0, row["coef"]], z=[0, 0],
            mode="lines+markers",
            line=dict(color=color, width=8),
            marker=dict(size=[3, 12], color=color,
                        line=dict(color="rgba(255,255,255,0.3)", width=1)),
            name=row["feature"].replace("_", " "),
            hovertemplate=f"<b>{row['feature'].replace('_',' ')}</b><br>Coefficient: {row['coef']:.4f}<extra></extra>",
            showlegend=False,
        ))

    fig_3d_coef.update_layout(
        scene=dict(
            xaxis=dict(
                tickvals=list(range(len(top20))),
                ticktext=[f[:15] for f in top20["feature"].str.replace("_", " ")],
                tickfont=dict(color="#9d9abf", size=8),
                gridcolor="rgba(45,43,69,0.5)",
                backgroundcolor="rgba(3,7,18,0.8)",
                showbackground=True,
                title=dict(text="Feature", font=dict(color="#9d9abf", size=11)),
            ),
            yaxis=dict(
                title=dict(text="Coefficient Value", font=dict(color="#9d9abf", size=11)),
                tickfont=dict(color="#9d9abf", size=9),
                gridcolor="rgba(45,43,69,0.5)",
                backgroundcolor="rgba(3,7,18,0.8)",
                showbackground=True,
                zerolinecolor="#7c3aed",
            ),
            zaxis=dict(showticklabels=False, showgrid=False,
                       backgroundcolor="rgba(0,0,0,0)"),
            bgcolor="rgba(3,7,18,0.9)",
            camera=dict(eye=dict(x=2.0, y=1.2, z=0.9)),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=10, b=10, l=0, r=0),
        height=480,
        font={"family": "Inter"},
    )
    st.markdown('<div class="chart-3d-wrap">', unsafe_allow_html=True)
    st.plotly_chart(fig_3d_coef, use_container_width=True, key="dashboard_coef_3d")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── 2D Coefficient Bar ──
    fig_coef2d = go.Figure(go.Bar(
        x=top20["coef"],
        y=top20["feature"].str.replace("_", " "),
        orientation="h",
        marker=dict(
            color=["#f87171" if c > 0 else "#22d3ee" for c in top20["coef"]],
            line=dict(color="rgba(0,0,0,0)"),
        ),
        hovertemplate="%{y}<br>Coefficient: %{x:.4f}<extra></extra>",
    ))
    fig_coef2d.update_layout(
        title=dict(text="Top 20 Feature Coefficients (2D View)",
                   font=dict(color="#d4d0f0", size=14, family="Space Grotesk")),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#9d9abf", "family": "Inter"},
        xaxis=dict(gridcolor="rgba(45,43,69,0.5)", zerolinecolor="#7c3aed",
                   title="Coefficient Value"),
        yaxis=dict(gridcolor="rgba(0,0,0,0)"),
        margin=dict(t=50, b=20, l=10, r=20), height=500,
    )
    st.plotly_chart(fig_coef2d, use_container_width=True, config={"displayModeBar": False}, key="dashboard_coef_2d")

    st.markdown("""
    <div class="advice-glass">
      <div class="advice-title">📖 How to Read These Charts</div>
      <div class="advice-item">
        <span class="advice-dot">🔴</span>
        <span><b style="color:#f87171">Red / Positive coefficients</b> increase the log-odds of churn — the larger the bar, the more influential the feature is in predicting churn.</span>
      </div>
      <div class="advice-item">
        <span class="advice-dot">🟢</span>
        <span><b style="color:#34d399">Green / Negative coefficients</b> decrease the log-odds of churn — these are protective factors that reduce churn probability.</span>
      </div>
      <div class="advice-item">
        <span class="advice-dot">ℹ️</span>
        <span>Features are one-hot encoded, so each bar represents a specific categorical value or numeric feature. Drag the 3D chart to explore from different angles.</span>
      </div>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════ TAB 6 — BATCH CSV ═══════════════════════════
with tab6:
    st.markdown('<div class="section-header"><span class="section-icon">📁</span><span class="section-title">Upload CSV & Batch Prediction</span></div>', unsafe_allow_html=True)

    sample_csv = df.head(20).to_csv(index=False).encode("utf-8")
    info_l, info_r = st.columns([1, 1])
    with info_l:
        st.markdown("""
        <div class="advice-glass">
          <div class="advice-title">CSV Requirements</div>
          <div class="advice-item"><span class="advice-dot">1</span><span>Use the same columns as the training data, including tenure, contract, payment method, and charges.</span></div>
          <div class="advice-item"><span class="advice-dot">2</span><span>Missing optional fields are filled with safe defaults so a partial file can still be scored.</span></div>
          <div class="advice-item"><span class="advice-dot">3</span><span>Download results as a CSV with churn probability, prediction, and risk tier.</span></div>
        </div>
        """, unsafe_allow_html=True)
    with info_r:
        st.download_button(
            "Download Sample CSV",
            data=sample_csv,
            file_name="churn_batch_sample.csv",
            mime="text/csv",
            use_container_width=True,
        )

    uploaded_file = st.file_uploader("Upload customer CSV", type=["csv"])
    if uploaded_file is not None:
        try:
            uploaded_df = pd.read_csv(uploaded_file)
            st.success(f"Loaded {len(uploaded_df):,} rows.")
            with st.spinner("Scoring customers..."):
                pred_df = batch_predict(uploaded_df)

            b1, b2, b3, b4 = st.columns(4)
            valid_probs = pd.to_numeric(pred_df["Churn_Prob_%"], errors="coerce")
            with b1:
                st.metric("Rows Scored", f"{len(pred_df):,}")
            with b2:
                st.metric("Avg Risk", f"{valid_probs.mean():.1f}%")
            with b3:
                st.metric("High/Critical", f"{pred_df['Risk_Tier'].isin(['High','Critical']).sum():,}")
            with b4:
                st.metric("Predicted Churn", f"{(pred_df['Prediction'] == 'Churn').sum():,}")

            st.dataframe(
                pred_df,
                use_container_width=True,
                height=420,
                column_config={
                    "Churn_Prob_%": st.column_config.ProgressColumn(
                        "Churn Prob %",
                        min_value=0,
                        max_value=100,
                        format="%.1f%%",
                    )
                },
            )
            st.download_button(
                "Download Batch Predictions",
                data=pred_df.to_csv(index=False).encode("utf-8"),
                file_name="churn_batch_predictions.csv",
                mime="text/csv",
                use_container_width=True,
            )
        except Exception as exc:
            st.error(f"Could not process this CSV: {exc}")
    else:
        st.info("Upload a customer CSV or download the sample template to test batch prediction.")


# ══════════════════════════════════ TAB 7 — WHY CHURN ═══════════════════════════
with tab7:
    st.markdown('<div class="section-header"><span class="section-icon">❓</span><span class="section-title">Why Churn Happens</span></div>', unsafe_allow_html=True)

    top_l, top_r = st.columns([1.15, 1])
    with top_l:
        st.plotly_chart(why_churn_chart(), use_container_width=True, config={"displayModeBar": False}, key="why_churn_chart")
    with top_r:
        st.markdown("""
        <div class="advice-glass">
          <div class="advice-title">How to Use This Section</div>
          <div class="advice-item"><span class="advice-dot">🎯</span><span>Compare each churn factor against the overall churn baseline.</span></div>
          <div class="advice-item"><span class="advice-dot">🛠️</span><span>Use the recommended fix cards below as ready-made retention playbooks.</span></div>
          <div class="advice-item"><span class="advice-dot">📣</span><span>Turn the highest bars into alert rules and campaign audiences.</span></div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    cols = st.columns(2)
    for i, item in enumerate(WHY_CHURN):
        with cols[i % 2]:
            st.markdown(f"""
            <div class="why-card" style="border-left-color:{item['color']}">
              <div class="why-title" style="color:{item['color']}">{item['icon']} {item['title']} · {item['stat']}</div>
              <div class="why-desc">{item['desc']}</div>
              <div style="font-size:.82rem;color:#d4d0f0;margin-top:.65rem">
                <b style="color:#22d3ee">Fix:</b> {item['fix']}
              </div>
            </div>
            """, unsafe_allow_html=True)


# ── Footer ──
st.markdown("<br><hr style='border-color:rgba(45,43,69,0.5);margin:1rem 0'>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center;color:#4b4869;font-size:.78rem;padding-bottom:1.5rem">
  🛡️ <b style="color:#7c3aed">ChurnShield AI</b> &nbsp;·&nbsp;
  Powered by Scikit-learn Logistic Regression &nbsp;·&nbsp;
  3D Visualizations by Plotly &nbsp;·&nbsp;
  Built with Streamlit
</div>
""", unsafe_allow_html=True)
