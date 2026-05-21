import streamlit as st
import pandas as pd
import numpy as np
import joblib
import warnings
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

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
</style>
""", unsafe_allow_html=True)


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
    predict_btn = st.button("⚡ Predict Churn Risk", use_container_width=True)

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
tab1, tab2, tab3 = st.tabs(["⚡  Prediction Engine", "🌐  3D Analytics", "🔬  Model Insights"])


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

        pos_idx      = np.argsort(contrib)[-5:][::-1]
        neg_idx      = np.argsort(contrib)[:5]
        risk_factors = [FEATURE_COLS[i].replace("_", " ") for i in pos_idx if contrib[i] > 0.001]
        prot_factors = [FEATURE_COLS[i].replace("_", " ") for i in neg_idx if contrib[i] < -0.001]

        # ── Row 1: Gauge | Result | Factors ──
        col_g, col_r, col_f = st.columns([1.1, 1.2, 1.7])

        with col_g:
            st.markdown('<div class="chart-3d-wrap">', unsafe_allow_html=True)
            st.plotly_chart(gauge_3d(prob), use_container_width=True, config={"displayModeBar": False})
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
        st.plotly_chart(feature_impact_3d(X), use_container_width=True, config={"displayModeBar": True})
        st.markdown('</div>', unsafe_allow_html=True)

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
            <div class="feature-item"><div class="feature-item-icon">💡</div><div class="feature-item-lbl">AI Advice</div></div>
            <div class="feature-item"><div class="feature-item-icon">🎯</div><div class="feature-item-lbl">Risk Tier</div></div>
            <div class="feature-item"><div class="feature-item-icon">📊</div><div class="feature-item-lbl">Key Factors</div></div>
          </div>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════ TAB 2 ════════════════════════════════════════
with tab2:
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
    st.plotly_chart(scatter_3d_churn(), use_container_width=True)
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
        st.plotly_chart(risk_surface_3d(), use_container_width=True)
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
        st.plotly_chart(bar_3d_churn(), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── 2D supporting charts ──
    st.markdown('<div class="section-header"><span class="section-icon">📊</span><span class="section-title">Distribution Analysis</span></div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Overall Churn Split**")
        st.plotly_chart(churn_donut(), use_container_width=True, config={"displayModeBar": False})
    with c2:
        st.markdown("**Tenure Distribution by Churn**")
        st.plotly_chart(tenure_hist(), use_container_width=True, config={"displayModeBar": False})

    c3, c4 = st.columns(2)
    with c3:
        st.markdown("**Churn by Contract Type**")
        st.plotly_chart(churn_by_col("Contract", "Contract"), use_container_width=True, config={"displayModeBar": False})
    with c4:
        st.markdown("**Monthly Charges Distribution**")
        st.plotly_chart(monthly_violin(), use_container_width=True, config={"displayModeBar": False})

    c5, c6 = st.columns(2)
    with c5:
        st.markdown("**Churn by Internet Service**")
        st.plotly_chart(churn_by_col("InternetService", "Internet Service"), use_container_width=True, config={"displayModeBar": False})
    with c6:
        st.markdown("**Churn by Payment Method**")
        st.plotly_chart(churn_by_col("PaymentMethod", "Payment Method"), use_container_width=True, config={"displayModeBar": False})

    # Raw data
    with st.expander("🔎 Explore Raw Dataset"):
        filter_churn = st.radio("Filter by Churn", ["All", "Yes", "No"], horizontal=True)
        show_df = df if filter_churn == "All" else df[df["Churn"] == filter_churn]
        st.dataframe(show_df.reset_index(drop=True), use_container_width=True, height=300)
        st.caption(f"Showing {len(show_df):,} of {len(df):,} records")


# ══════════════════════════════════ TAB 3 ════════════════════════════════════════
with tab3:
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
    st.plotly_chart(fig_3d_coef, use_container_width=True)
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
    st.plotly_chart(fig_coef2d, use_container_width=True, config={"displayModeBar": False})

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
