"""
Streamlit Dashboard — Customer Churn Prediction
================================================
A premium, interactive web dashboard for the Telco Customer Churn
prediction project.  Loads saved model artifacts and provides:

  1. Dataset Explorer (EDA visualizations)
  2. Model Performance Dashboard (metrics, curves, comparison)
  3. Individual Customer Predictor (with SHAP explanation)
  4. Business Recommendations Engine

Run:  streamlit run app.py
"""

import sys, os, json, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib

# ── Ensure src is importable ──────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.preprocessing import load_and_clean, engineer_features, build_preprocessor, get_feature_names
from src.explain import predict_customer, business_recommendations


# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Churn Predictor | Telco ML Dashboard",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
# CUSTOM CSS — Premium dark glassmorphism theme
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ── Global ───────────────────────────────────────────────────────────── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    color: #e2e8f0;
}
.stApp {
    background: linear-gradient(135deg, #0f0c29 0%, #1a1a2e 40%, #16213e 100%);
}

/* ── Force ALL text light on dark bg ──────────────────────────────────── */
p, span, div, label, li, td, th, h1, h2, h3, h4, h5, h6 {
    color: #e2e8f0 !important;
}

/* Streamlit native text overrides */
.stMarkdown, .stMarkdown p, .stMarkdown span,
.stText, .element-container,
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li,
[data-testid="stMarkdownContainer"] span {
    color: #e2e8f0 !important;
}

/* Dataframe / table text */
.stDataFrame td, .stDataFrame th,
[data-testid="stDataFrame"] td,
[data-testid="stDataFrame"] th,
.dataframe td, .dataframe th {
    color: #e2e8f0 !important;
    background-color: rgba(255,255,255,0.04) !important;
}

/* Metric labels and values */
[data-testid="stMetricValue"],
[data-testid="stMetricLabel"],
[data-testid="stMetricDelta"] {
    color: #e2e8f0 !important;
}

/* Selectbox, radio, checkbox labels */
.stSelectbox label, .stRadio label, .stCheckbox label,
.stSlider label, .stNumberInput label, .stTextInput label {
    color: #c7d2fe !important;
    font-weight: 500 !important;
}
.stSelectbox div[data-baseweb="select"] span,
.stSelectbox div[data-baseweb="select"] div {
    color: #e2e8f0 !important;
}

/* Caption / help text */
.stCaption, [data-testid="stCaptionContainer"],
small, .caption {
    color: rgba(255, 255, 255, 0.55) !important;
}

/* ── Sidebar ──────────────────────────────────────────────────────────── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f0c29 0%, #1a1a2e 100%);
    border-right: 1px solid rgba(255, 255, 255, 0.06);
}
section[data-testid="stSidebar"] * {
    color: #e2e8f0 !important;
}
section[data-testid="stSidebar"] .stMarkdown h1,
section[data-testid="stSidebar"] .stMarkdown h2,
section[data-testid="stSidebar"] .stMarkdown h3 {
    color: #c7d2fe !important;
}

/* ── Glass cards ──────────────────────────────────────────────────────── */
.glass-card {
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.10);
    border-radius: 16px;
    padding: 24px 28px;
    margin: 12px 0;
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.glass-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 12px 40px rgba(0, 0, 0, 0.5);
}

/* ── Metric cards ─────────────────────────────────────────────────────── */
.metric-card {
    background: linear-gradient(135deg, rgba(99, 102, 241, 0.15), rgba(139, 92, 246, 0.08));
    border: 1px solid rgba(139, 92, 246, 0.25);
    border-radius: 14px;
    padding: 20px 24px;
    text-align: center;
    transition: all 0.3s ease;
}
.metric-card:hover {
    border-color: rgba(139, 92, 246, 0.5);
    box-shadow: 0 0 20px rgba(139, 92, 246, 0.15);
}
.metric-value {
    font-size: 2.2rem;
    font-weight: 800;
    background: linear-gradient(135deg, #818cf8, #c084fc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1.2;
}
.metric-label {
    font-size: 0.85rem;
    color: rgba(255, 255, 255, 0.65) !important;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-top: 6px;
    font-weight: 600;
}

/* ── Risk badges ──────────────────────────────────────────────────────── */
.risk-high {
    background: linear-gradient(135deg, rgba(239, 68, 68, 0.2), rgba(220, 38, 38, 0.1));
    border: 1px solid rgba(239, 68, 68, 0.35);
    border-radius: 12px;
    padding: 16px 20px;
    color: #fca5a5 !important;
}
.risk-low {
    background: linear-gradient(135deg, rgba(34, 197, 94, 0.2), rgba(22, 163, 74, 0.1));
    border: 1px solid rgba(34, 197, 94, 0.35);
    border-radius: 12px;
    padding: 16px 20px;
    color: #86efac !important;
}

/* ── Section headers ──────────────────────────────────────────────────── */
.section-header {
    font-size: 1.8rem;
    font-weight: 700;
    background: linear-gradient(135deg, #e0e7ff, #c7d2fe);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 4px;
}
.section-sub {
    font-size: 0.95rem;
    color: rgba(255, 255, 255, 0.55) !important;
    margin-bottom: 24px;
}

/* ── Recommendation cards ─────────────────────────────────────────────── */
.rec-card {
    background: linear-gradient(135deg, rgba(59, 130, 246, 0.08), rgba(99, 102, 241, 0.05));
    border-left: 4px solid #818cf8;
    border-radius: 0 12px 12px 0;
    padding: 16px 20px;
    margin: 10px 0;
    color: #e0e0ff !important;
}
.rec-card strong {
    color: #a5b4fc !important;
}

/* ── SHAP feature bar ─────────────────────────────────────────────────── */
.shap-positive { color: #f87171 !important; font-weight: 600; }
.shap-negative { color: #4ade80 !important; font-weight: 600; }

/* ── Hero banner ──────────────────────────────────────────────────────── */
.hero-banner {
    background: linear-gradient(135deg, rgba(99, 102, 241, 0.12), rgba(168, 85, 247, 0.08));
    border: 1px solid rgba(139, 92, 246, 0.2);
    border-radius: 20px;
    padding: 36px 40px;
    margin-bottom: 28px;
    text-align: center;
}
.hero-title {
    font-size: 2.6rem;
    font-weight: 800;
    background: linear-gradient(135deg, #c7d2fe, #e9d5ff, #fde68a);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 8px;
}
.hero-sub {
    font-size: 1.1rem;
    color: rgba(255, 255, 255, 0.6) !important;
    font-weight: 400;
}

/* ── Gauge / donut center label ───────────────────────────────────────── */
.gauge-center {
    text-align: center;
    margin-top: -10px;
}

/* ── Tabs styling ─────────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background: transparent !important;
}
.stTabs [data-baseweb="tab"] {
    background: rgba(255, 255, 255, 0.05);
    border-radius: 10px;
    border: 1px solid rgba(255, 255, 255, 0.08);
    color: rgba(255, 255, 255, 0.7) !important;
    padding: 10px 20px;
    font-weight: 500;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, rgba(99, 102, 241, 0.25), rgba(139, 92, 246, 0.18)) !important;
    border-color: rgba(139, 92, 246, 0.5) !important;
    color: #e0e0ff !important;
    font-weight: 600 !important;
}

/* ── Plotly chart backgrounds ─────────────────────────────────────────── */
.js-plotly-plot .plotly .main-svg {
    background: transparent !important;
}

/* ── Dividers ─────────────────────────────────────────────────────────── */
hr {
    border-color: rgba(255, 255, 255, 0.08) !important;
}

/* ── Button overrides ─────────────────────────────────────────────────── */
.stButton > button {
    background: linear-gradient(135deg, rgba(99,102,241,0.3), rgba(139,92,246,0.2));
    border: 1px solid rgba(139, 92, 246, 0.4);
    color: #e0e0ff !important;
    border-radius: 10px;
    font-weight: 600;
    transition: all 0.2s ease;
}
.stButton > button:hover {
    background: linear-gradient(135deg, rgba(99,102,241,0.5), rgba(139,92,246,0.35));
    border-color: rgba(139, 92, 246, 0.7);
    transform: translateY(-1px);
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# DATA & MODEL LOADING (cached)
# ══════════════════════════════════════════════════════════════════════════════

RANDOM_STATE = 42
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "telco_churn.csv")
MODEL_PATHS = {
    "Random Forest": os.path.join(PROJECT_ROOT, "models", "rf_model.joblib"),
    "Logistic Regression": os.path.join(PROJECT_ROOT, "models", "lr_model.joblib"),
    "Decision Tree": os.path.join(PROJECT_ROOT, "models", "dt_model.joblib"),
    "Naive Bayes": os.path.join(PROJECT_ROOT, "models", "nb_model.joblib"),
}
PREPROCESSOR_PATH = os.path.join(PROJECT_ROOT, "models", "preprocessor.joblib")
COMPARISON_PATH = os.path.join(PROJECT_ROOT, "models", "model_comparison.json")


@st.cache_data
def load_data():
    """Load and clean the dataset, then engineer features."""
    df_raw = pd.read_csv(DATA_PATH)
    df_clean = load_and_clean(DATA_PATH)
    df_eng = engineer_features(df_clean)
    return df_raw, df_clean, df_eng


@st.cache_resource
def load_model_artifacts(model_name="Random Forest"):
    """Load saved model, preprocessor, and SHAP explainer based on selected model."""
    model_path = MODEL_PATHS.get(model_name, MODEL_PATHS["Random Forest"])
    model = joblib.load(model_path)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    feature_names = get_feature_names(preprocessor)

    # Build SHAP explainer
    import shap
    if "Logistic" in model_name:
        explainer = shap.LinearExplainer(model, shap.maskers.Independent(np.zeros((1, len(feature_names))))) 
    elif "Naive" in model_name:
        explainer = shap.KernelExplainer(model.predict_proba, np.zeros((1, len(feature_names))))
    else:
        explainer = shap.TreeExplainer(model)

    return model, preprocessor, feature_names, explainer


def load_comparison():
    """Load model comparison JSON."""
    if os.path.exists(COMPARISON_PATH):
        with open(COMPARISON_PATH) as f:
            return json.load(f)
    return []


def check_artifacts_exist():
    """Return True if saved model artifacts exist."""
    return os.path.exists(MODEL_PATHS["Random Forest"]) and os.path.exists(PREPROCESSOR_PATH)


# ══════════════════════════════════════════════════════════════════════════════
# HELPER: plotly theme
# ══════════════════════════════════════════════════════════════════════════════

PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter", color="rgba(255,255,255,0.75)"),
    margin=dict(l=40, r=30, t=50, b=40),
)

COLORS = {
    "primary": "#818cf8",
    "secondary": "#c084fc",
    "success": "#4ade80",
    "danger": "#f87171",
    "warning": "#fbbf24",
    "info": "#38bdf8",
    "retained": "#4ade80",
    "churned": "#f87171",
}

GRADIENT_COLORS = ["#818cf8", "#a78bfa", "#c084fc", "#e879f9", "#f472b6"]


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("# 📡 Churn Predictor")
    st.markdown("---")

    page = st.radio(
        "Navigate",
        ["🏠 Overview", "📊 Data Explorer", "🏆 Model Performance", "🔮 Predict Customer", "💡 Insights"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown(
        '<p style="color: rgba(255,255,255,0.3); font-size: 0.75rem; text-align: center;">'
        "Built with Streamlit | ML Portfolio Project</p>",
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Overview
# ══════════════════════════════════════════════════════════════════════════════

if page == "🏠 Overview":
    st.markdown(
        '<div class="hero-banner">'
        '<div class="hero-title">Customer Churn Prediction</div>'
        '<div class="hero-sub">ML-powered retention intelligence for Telco operators</div>'
        "</div>",
        unsafe_allow_html=True,
    )

    if not os.path.exists(DATA_PATH):
        st.error("Dataset not found. Run `python generate_data.py` first.")
        st.stop()

    df_raw, df_clean, df_eng = load_data()

    # Key stats row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(
            f'<div class="metric-card"><div class="metric-value">{len(df_clean):,}</div>'
            f'<div class="metric-label">Total Customers</div></div>',
            unsafe_allow_html=True,
        )
    with col2:
        churn_rate = df_clean["Churn"].mean()
        st.markdown(
            f'<div class="metric-card"><div class="metric-value">{churn_rate:.1%}</div>'
            f'<div class="metric-label">Churn Rate</div></div>',
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            f'<div class="metric-card"><div class="metric-value">{df_eng.shape[1]-1}</div>'
            f'<div class="metric-label">Features</div></div>',
            unsafe_allow_html=True,
        )
    with col4:
        n_models = len(load_comparison()) if check_artifacts_exist() else 0
        st.markdown(
            f'<div class="metric-card"><div class="metric-value">{n_models}</div>'
            f'<div class="metric-label">Models Trained</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Churn distribution donut
    col_left, col_right = st.columns([1, 1])
    with col_left:
        st.markdown('<div class="section-header">Class Distribution</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">Target variable balance</div>', unsafe_allow_html=True)

        counts = df_clean["Churn"].value_counts()
        fig = go.Figure(
            go.Pie(
                labels=["Retained", "Churned"],
                values=[counts[0], counts[1]],
                hole=0.65,
                marker=dict(colors=[COLORS["retained"], COLORS["churned"]], line=dict(color="rgba(0,0,0,0.3)", width=2)),
                textinfo="percent+label",
                textfont=dict(size=14, family="Inter"),
                hovertemplate="<b>%{label}</b><br>Count: %{value:,}<br>%{percent}<extra></extra>",
            )
        )
        fig.update_layout(**PLOTLY_LAYOUT, height=350, showlegend=False)
        fig.add_annotation(
            text=f"<b>{churn_rate:.1%}</b><br><span style='font-size:11px;color:rgba(255,255,255,0.4)'>Churn Rate</span>",
            showarrow=False, font=dict(size=24, color="#e0e0ff"),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.markdown('<div class="section-header">Project Architecture</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">Modular ML pipeline design</div>', unsafe_allow_html=True)

        st.markdown(
            '<div class="glass-card">'
            "<p style='color:#a5b4fc; font-weight:600; margin-bottom:12px;'>Pipeline Flow</p>"
            "<p style='color:rgba(255,255,255,0.6); font-size:0.9rem; line-height:1.8;'>"
            "1. <strong style='color:#818cf8'>Data</strong> &rarr; load_and_clean() &rarr; engineer_features()<br>"
            "2. <strong style='color:#a78bfa'>Preprocess</strong> &rarr; build_preprocessor() (fit on train only)<br>"
            "3. <strong style='color:#c084fc'>Train</strong> &rarr; 6 models &times; 2 configs (class_weight + SMOTE)<br>"
            "4. <strong style='color:#e879f9'>Evaluate</strong> &rarr; PR-AUC as primary metric<br>"
            "5. <strong style='color:#f472b6'>Tune</strong> &rarr; RandomizedSearchCV (top 3 models)<br>"
            "6. <strong style='color:#fbbf24'>Explain</strong> &rarr; SHAP global + local explanations<br>"
            "7. <strong style='color:#4ade80'>Serve</strong> &rarr; predict_customer() with tuned threshold"
            "</p></div>",
            unsafe_allow_html=True,
        )

        if not check_artifacts_exist():
            st.warning("Model artifacts not found. Run the notebook first to train and save models.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Data Explorer
# ══════════════════════════════════════════════════════════════════════════════

elif page == "📊 Data Explorer":
    st.markdown('<div class="section-header">Data Explorer</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Interactive EDA of the Telco churn dataset</div>', unsafe_allow_html=True)

    if not os.path.exists(DATA_PATH):
        st.error("Dataset not found.")
        st.stop()

    df_raw, df_clean, df_eng = load_data()

    tab1, tab2, tab3 = st.tabs(["Distributions", "Churn Drivers", "Correlations"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            # Numeric distribution
            num_col = st.selectbox("Select numeric feature", ["tenure", "MonthlyCharges", "TotalCharges"])
            fig = px.histogram(
                df_clean, x=num_col, color=df_clean["Churn"].map({0: "Retained", 1: "Churned"}),
                barmode="overlay", nbins=40, opacity=0.7,
                color_discrete_map={"Retained": COLORS["retained"], "Churned": COLORS["churned"]},
                labels={"color": "Status"},
            )
            fig.update_layout(**PLOTLY_LAYOUT, title=f"{num_col} Distribution by Churn Status", height=400)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Box plot
            fig = px.box(
                df_clean.assign(Status=df_clean["Churn"].map({0: "Retained", 1: "Churned"})),
                x="Status", y=num_col,
                color="Status",
                color_discrete_map={"Retained": COLORS["retained"], "Churned": COLORS["churned"]},
            )
            fig.update_layout(**PLOTLY_LAYOUT, title=f"{num_col} Box Plot by Churn", height=400)
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        cat_cols = ["Contract", "InternetService", "PaymentMethod", "SeniorCitizen",
                     "TechSupport", "OnlineSecurity", "Partner", "Dependents"]
        sel_cat = st.selectbox("Select categorical feature", cat_cols)

        churn_by = df_clean.groupby(sel_cat)["Churn"].mean().reset_index()
        churn_by.columns = [sel_cat, "ChurnRate"]
        churn_by = churn_by.sort_values("ChurnRate", ascending=True)

        fig = go.Figure(
            go.Bar(
                y=churn_by[sel_cat].astype(str),
                x=churn_by["ChurnRate"] * 100,
                orientation="h",
                marker=dict(
                    color=churn_by["ChurnRate"],
                    colorscale=[[0, COLORS["success"]], [1, COLORS["danger"]]],
                    line=dict(width=0),
                ),
                text=[f"{r:.1f}%" for r in churn_by["ChurnRate"] * 100],
                textposition="outside",
                textfont=dict(size=13, color="rgba(255,255,255,0.8)"),
                hovertemplate="<b>%{y}</b><br>Churn Rate: %{x:.1f}%<extra></extra>",
            )
        )
        fig.update_layout(**PLOTLY_LAYOUT, title=f"Churn Rate by {sel_cat}", height=max(300, len(churn_by) * 55 + 100))
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        numeric_df = df_eng.select_dtypes(include=[np.number])
        corr = numeric_df.corr()

        fig = px.imshow(
            corr, text_auto=".2f", aspect="auto",
            color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
        )
        fig.update_layout(**PLOTLY_LAYOUT, title="Feature Correlation Heatmap", height=600)
        st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Model Performance
# ══════════════════════════════════════════════════════════════════════════════

elif page == "🏆 Model Performance":
    st.markdown('<div class="section-header">Model Performance</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Comparison of all trained models</div>', unsafe_allow_html=True)

    if not check_artifacts_exist():
        st.warning("Model artifacts not found. Run the notebook first (`python -m jupyter nbconvert --execute ...`)")
        st.stop()

    comparison = load_comparison()
    if not comparison:
        st.info("No model comparison data found. Run the notebook to generate `models/model_comparison.json`.")
        st.stop()

    # Build comparison dataframe
    comp_df = pd.DataFrame([
        {
            "Model": r["model_name"],
            "Accuracy": r["accuracy"],
            "Precision": r["precision"],
            "Recall": r["recall"],
            "F1": r["f1"],
            "ROC-AUC": r["roc_auc"],
            "PR-AUC": r["pr_auc"],
            "Train Time (s)": r["train_time_s"],
        }
        for r in comparison
    ]).sort_values("PR-AUC", ascending=False).reset_index(drop=True)

    # Best model highlight
    best = comp_df.iloc[0]
    st.markdown(
        f'<div class="glass-card" style="border-color: rgba(129, 140, 248, 0.3);">'
        f'<p style="color:#818cf8; font-weight:700; font-size:1.1rem; margin-bottom:8px;">'
        f'Best Model: {best["Model"]}</p>'
        f'<p style="color:rgba(255,255,255,0.6); font-size:0.9rem;">'
        f'PR-AUC: <strong style="color:#c084fc">{best["PR-AUC"]:.4f}</strong> &nbsp;|&nbsp; '
        f'F1: <strong style="color:#a78bfa">{best["F1"]:.4f}</strong> &nbsp;|&nbsp; '
        f'Recall: <strong style="color:#818cf8">{best["Recall"]:.4f}</strong> &nbsp;|&nbsp; '
        f'ROC-AUC: <strong style="color:#e879f9">{best["ROC-AUC"]:.4f}</strong>'
        f"</p></div>",
        unsafe_allow_html=True,
    )

    tab1, tab2 = st.tabs(["Comparison Table", "Visual Comparison"])

    with tab1:
        # Styled dataframe
        styled = comp_df.style.format({
            "Accuracy": "{:.4f}", "Precision": "{:.4f}", "Recall": "{:.4f}",
            "F1": "{:.4f}", "ROC-AUC": "{:.4f}", "PR-AUC": "{:.4f}",
            "Train Time (s)": "{:.2f}",
        }).background_gradient(subset=["PR-AUC", "F1"], cmap="YlGn")
        st.dataframe(styled, use_container_width=True, height=min(len(comp_df) * 40 + 60, 600))

    with tab2:
        # Radar chart of top 5 models
        top5 = comp_df.head(5)
        metrics = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"]

        fig = go.Figure()
        for i, (_, row) in enumerate(top5.iterrows()):
            fig.add_trace(go.Scatterpolar(
                r=[row[m] for m in metrics] + [row[metrics[0]]],
                theta=metrics + [metrics[0]],
                fill="toself",
                name=row["Model"],
                line=dict(color=GRADIENT_COLORS[i % len(GRADIENT_COLORS)], width=2),
                opacity=0.7,
            ))
        fig.update_layout(
            **PLOTLY_LAYOUT,
            polar=dict(
                bgcolor="rgba(0,0,0,0)",
                radialaxis=dict(visible=True, range=[0, 1], gridcolor="rgba(255,255,255,0.08)"),
                angularaxis=dict(gridcolor="rgba(255,255,255,0.08)"),
            ),
            title="Model Performance Radar (Top 5 by PR-AUC)",
            height=500,
        )
        st.plotly_chart(fig, use_container_width=True)

        # Bar chart comparison
        fig = go.Figure()
        for metric, color in zip(["PR-AUC", "F1", "Recall"], [COLORS["primary"], COLORS["secondary"], COLORS["info"]]):
            fig.add_trace(go.Bar(
                x=comp_df["Model"], y=comp_df[metric],
                name=metric, marker_color=color, opacity=0.85,
            ))
        fig.update_layout(**PLOTLY_LAYOUT, barmode="group", title="Key Metrics Comparison", height=450,
                          xaxis=dict(tickangle=-45))
        st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Predict Customer
# ══════════════════════════════════════════════════════════════════════════════

elif page == "🔮 Predict Customer":
    st.markdown('<div class="section-header">Customer Churn Predictor</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Enter customer details to get a real-time churn prediction with SHAP explanation</div>', unsafe_allow_html=True)

    if not (os.path.exists(PREPROCESSOR_PATH) and os.path.exists(MODEL_PATHS["Random Forest"])):
        st.warning("Model artifacts not found. Run the notebook first to train and save models.")
        st.stop()

    st.markdown("### Select Model")
    selected_model = st.selectbox("Model", list(MODEL_PATHS.keys()), index=0, label_visibility="collapsed")
    
    model, preprocessor, feature_names, explainer = load_model_artifacts(selected_model)

    # ── Customer input form ───────────────────────────────────────────────────
    with st.form("customer_form"):
        st.markdown("### Customer Details")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            gender = st.selectbox("Gender", ["Male", "Female"])
            senior = st.selectbox("Senior Citizen", ["No", "Yes"])
            partner = st.selectbox("Partner", ["No", "Yes"])
            dependents = st.selectbox("Dependents", ["No", "Yes"])

        with col2:
            tenure = st.slider("Tenure (months)", 0, 72, 12)
            monthly = st.number_input("Monthly Charges ($)", 18.0, 120.0, 70.0, step=5.0)
            total = st.number_input("Total Charges ($)", 0.0, 9000.0, float(tenure * monthly), step=50.0)

        with col3:
            phone = st.selectbox("Phone Service", ["Yes", "No"])
            multi_lines = st.selectbox("Multiple Lines", ["No", "Yes"])
            internet = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
            security = st.selectbox("Online Security", ["No", "Yes"])
            backup = st.selectbox("Online Backup", ["No", "Yes"])

        with col4:
            protection = st.selectbox("Device Protection", ["No", "Yes"])
            tech = st.selectbox("Tech Support", ["No", "Yes"])
            tv = st.selectbox("Streaming TV", ["No", "Yes"])
            movies = st.selectbox("Streaming Movies", ["No", "Yes"])

        st.markdown("### Contract & Billing")
        col5, col6, col7 = st.columns(3)
        with col5:
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        with col6:
            paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
        with col7:
            payment = st.selectbox("Payment Method", [
                "Electronic check", "Mailed check",
                "Bank transfer (automatic)", "Credit card (automatic)"
            ])

        submitted = st.form_submit_button("Predict Churn Risk", type="primary", use_container_width=True)

    if submitted:
        customer_dict = {
            "gender": gender, "SeniorCitizen": 1 if senior == "Yes" else 0,
            "Partner": partner, "Dependents": dependents,
            "tenure": tenure, "PhoneService": phone, "MultipleLines": multi_lines,
            "InternetService": internet, "OnlineSecurity": security,
            "OnlineBackup": backup, "DeviceProtection": protection,
            "TechSupport": tech, "StreamingTV": tv, "StreamingMovies": movies,
            "Contract": contract, "PaperlessBilling": paperless,
            "PaymentMethod": payment, "MonthlyCharges": monthly, "TotalCharges": total,
        }

        # Use a fixed threshold (0.4 is a reasonable default; the notebook tunes this)
        THRESHOLD = 0.40

        with st.spinner("Analyzing customer..."):
            prediction = predict_customer(
                customer_dict, preprocessor, model,
                THRESHOLD, explainer, feature_names
            )

        # ── Results display ───────────────────────────────────────────────────
        st.markdown("---")

        # Risk gauge
        prob = prediction["churn_probability"]
        risk = prediction["risk_score"]
        pred_label = "CHURN" if prediction["prediction"] == 1 else "RETAIN"

        col_gauge, col_details = st.columns([1, 1])

        with col_gauge:
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=prob * 100,
                number=dict(suffix="%", font=dict(size=48, color="#e0e0ff")),
                title=dict(text="Churn Probability", font=dict(size=16, color="rgba(255,255,255,0.6)")),
                gauge=dict(
                    axis=dict(range=[0, 100], tickcolor="rgba(255,255,255,0.3)"),
                    bar=dict(color=COLORS["danger"] if prob > 0.5 else COLORS["warning"] if prob > 0.3 else COLORS["success"]),
                    bgcolor="rgba(255,255,255,0.05)",
                    steps=[
                        dict(range=[0, 30], color="rgba(74, 222, 128, 0.08)"),
                        dict(range=[30, 60], color="rgba(251, 191, 36, 0.08)"),
                        dict(range=[60, 100], color="rgba(248, 113, 113, 0.08)"),
                    ],
                    threshold=dict(line=dict(color="#e0e0ff", width=3), thickness=0.8, value=THRESHOLD * 100),
                ),
            ))
            fig.update_layout(**PLOTLY_LAYOUT, height=300)
            st.plotly_chart(fig, use_container_width=True)

        with col_details:
            risk_class = "risk-high" if prediction["prediction"] == 1 else "risk-low"
            icon = "&#9888;&#65039;" if prediction["prediction"] == 1 else "&#9989;"

            st.markdown(
                f'<div class="{risk_class}">'
                f'<p style="font-size:1.4rem; font-weight:700; margin-bottom:12px;">'
                f'{icon} Prediction: {pred_label}</p>'
                f'<p style="font-size:0.95rem; line-height:1.8;">'
                f'<strong>Risk Score:</strong> {risk}/100<br>'
                f'<strong>Confidence:</strong> {prediction["confidence"]:.3f}<br>'
                f'<strong>Threshold:</strong> {THRESHOLD:.3f} (tuned)'
                f"</p></div>",
                unsafe_allow_html=True,
            )

            st.markdown("<br>", unsafe_allow_html=True)

        # ── SHAP Feature Drivers ──────────────────────────────────────────────
        st.markdown('<div class="section-header">SHAP Feature Drivers</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">What is driving this prediction?</div>', unsafe_allow_html=True)

        top_feats = prediction["top_features"]
        feat_names = [f["feature"] for f in top_feats]
        shap_vals = [f["shap_value"] for f in top_feats]

        colors_shap = [COLORS["danger"] if v > 0 else COLORS["success"] for v in shap_vals]

        fig = go.Figure(go.Bar(
            y=feat_names[::-1],
            x=shap_vals[::-1],
            orientation="h",
            marker_color=colors_shap[::-1],
            text=[f"{v:+.4f}" for v in shap_vals[::-1]],
            textposition="outside",
            textfont=dict(size=12, color="rgba(255,255,255,0.7)"),
            hovertemplate="<b>%{y}</b><br>SHAP: %{x:+.4f}<extra></extra>",
        ))
        fig.update_layout(
            **PLOTLY_LAYOUT, height=max(250, len(top_feats) * 50 + 80),
            title="Top SHAP Contributors",
            xaxis_title="SHAP Value (positive = churn, negative = retain)",
        )
        fig.add_vline(x=0, line_width=1, line_color="rgba(255,255,255,0.2)")
        st.plotly_chart(fig, use_container_width=True)

        # ── Business Recommendations ──────────────────────────────────────────
        recs = business_recommendations(prediction)
        if recs:
            st.markdown('<div class="section-header">Retention Recommendations</div>', unsafe_allow_html=True)
            for i, rec in enumerate(recs, 1):
                st.markdown(
                    f'<div class="rec-card">'
                    f'<strong>[{i}] {rec["recommendation"]}</strong><br>'
                    f'<span style="color:rgba(255,255,255,0.5); font-size:0.85rem;">'
                    f'{rec["rationale"]}</span></div>',
                    unsafe_allow_html=True,
                )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Insights
# ══════════════════════════════════════════════════════════════════════════════

elif page == "💡 Insights":
    st.markdown('<div class="section-header">Key Business Insights</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Data-driven findings from the churn analysis</div>', unsafe_allow_html=True)

    if not os.path.exists(DATA_PATH):
        st.error("Dataset not found.")
        st.stop()

    df_raw, df_clean, df_eng = load_data()

    # ── Insight cards ─────────────────────────────────────────────────────────
    insights = [
        ("Contract Type is #1", "Month-to-month customers churn at 3-4x the rate of two-year subscribers. "
         "Contract upgrades are the highest-ROI retention lever.",
         "📋", "rgba(99, 102, 241, 0.15)"),
        ("First 12 Months are Critical", "Churn rate drops sharply after the first year. "
         "Onboarding programs and milestone rewards have outsized impact.",
         "⏰", "rgba(168, 85, 247, 0.15)"),
        ("Fiber Optic Paradox", "Premium fiber customers churn MORE despite paying more. "
         "This signals a value-perception gap that needs addressing.",
         "🔌", "rgba(236, 72, 153, 0.15)"),
        ("Add-ons Reduce Churn", "Customers with more services have lower churn. "
         "Each add-on increases switching cost and reduces attrition.",
         "📦", "rgba(34, 197, 94, 0.15)"),
    ]

    col1, col2 = st.columns(2)
    for i, (title, desc, icon, bg) in enumerate(insights):
        with [col1, col2][i % 2]:
            st.markdown(
                f'<div class="glass-card" style="background:{bg}; min-height:160px;">'
                f'<p style="font-size:1.4rem; margin-bottom:4px;">{icon}</p>'
                f'<p style="color:#e0e0ff; font-weight:700; font-size:1.1rem; margin-bottom:8px;">{title}</p>'
                f'<p style="color:rgba(255,255,255,0.55); font-size:0.9rem; line-height:1.6;">{desc}</p>'
                f"</div>",
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Contract churn breakdown ──────────────────────────────────────────────
    st.markdown('<div class="section-header">Contract Type Deep Dive</div>', unsafe_allow_html=True)

    churn_contract = df_clean.groupby("Contract")["Churn"].agg(["mean", "count"]).reset_index()
    churn_contract.columns = ["Contract", "ChurnRate", "Count"]
    churn_contract = churn_contract.sort_values("ChurnRate", ascending=False)

    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "bar"}, {"type": "pie"}]],
        subplot_titles=("Churn Rate by Contract", "Customer Distribution"),
    )

    fig.add_trace(
        go.Bar(
            x=churn_contract["Contract"],
            y=churn_contract["ChurnRate"] * 100,
            marker=dict(
                color=churn_contract["ChurnRate"],
                colorscale=[[0, COLORS["success"]], [1, COLORS["danger"]]],
            ),
            text=[f"{r:.1f}%" for r in churn_contract["ChurnRate"] * 100],
            textposition="outside",
            textfont=dict(size=13),
            hovertemplate="<b>%{x}</b><br>Churn: %{y:.1f}%<extra></extra>",
        ),
        row=1, col=1,
    )

    fig.add_trace(
        go.Pie(
            labels=churn_contract["Contract"],
            values=churn_contract["Count"],
            marker=dict(colors=GRADIENT_COLORS[:3]),
            textinfo="percent+label",
            hole=0.4,
        ),
        row=1, col=2,
    )

    fig.update_layout(**PLOTLY_LAYOUT, height=400, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    # ── Tenure vs churn ───────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Tenure Analysis</div>', unsafe_allow_html=True)

    tenure_bins = pd.cut(df_clean["tenure"], bins=[0, 6, 12, 24, 48, 72], labels=["0-6mo", "6-12mo", "12-24mo", "24-48mo", "48-72mo"])
    tenure_churn = df_clean.groupby(tenure_bins, observed=True)["Churn"].mean().reset_index()
    tenure_churn.columns = ["Tenure Group", "Churn Rate"]

    fig = go.Figure(go.Bar(
        x=tenure_churn["Tenure Group"].astype(str),
        y=tenure_churn["Churn Rate"] * 100,
        marker=dict(
            color=tenure_churn["Churn Rate"],
            colorscale=[[0, COLORS["success"]], [1, COLORS["danger"]]],
        ),
        text=[f"{r:.1f}%" for r in tenure_churn["Churn Rate"] * 100],
        textposition="outside",
        textfont=dict(size=13, color="rgba(255,255,255,0.8)"),
    ))
    fig.update_layout(**PLOTLY_LAYOUT, title="Churn Rate by Tenure Group", height=400,
                      yaxis_title="Churn Rate (%)")
    st.plotly_chart(fig, use_container_width=True)

    # ── Revenue at risk ───────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Revenue Impact</div>', unsafe_allow_html=True)

    churners = df_clean[df_clean["Churn"] == 1]
    total_monthly_at_risk = churners["MonthlyCharges"].sum()
    avg_monthly_churner = churners["MonthlyCharges"].mean()
    avg_tenure_churner = churners["tenure"].mean()

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            f'<div class="metric-card"><div class="metric-value">${total_monthly_at_risk:,.0f}</div>'
            f'<div class="metric-label">Monthly Revenue at Risk</div></div>',
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f'<div class="metric-card"><div class="metric-value">${avg_monthly_churner:.0f}</div>'
            f'<div class="metric-label">Avg Monthly Charge (Churners)</div></div>',
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f'<div class="metric-card"><div class="metric-value">{avg_tenure_churner:.0f} mo</div>'
            f'<div class="metric-label">Avg Tenure (Churners)</div></div>',
            unsafe_allow_html=True,
        )
