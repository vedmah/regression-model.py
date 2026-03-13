import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.datasets import fetch_california_housing, load_diabetes
import warnings
warnings.filterwarnings('ignore')

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Linear Regression Lab",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global styles ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600;700&display=swap');

/* Base */
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: #0b0f1a; color: #e2e8f0; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: #111827;
    border-right: 1px solid #1f2d3d;
}
[data-testid="stSidebar"] * { color: #cbd5e1 !important; }
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label { color: #94a3b8 !important; font-size: 0.82rem; text-transform: uppercase; letter-spacing: .06em; }

/* Headers */
h1 { font-family: 'Space Mono', monospace !important; font-size: 2rem !important; color: #38bdf8 !important; letter-spacing: -0.03em; }
h2, h3 { font-family: 'Space Mono', monospace !important; color: #7dd3fc !important; }

/* Metric cards */
.metric-card {
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    text-align: center;
    transition: transform .2s, box-shadow .2s;
}
.metric-card:hover { transform: translateY(-3px); box-shadow: 0 8px 24px rgba(56,189,248,.15); }
.metric-label { font-size: .75rem; text-transform: uppercase; letter-spacing: .1em; color: #64748b; margin-bottom: .3rem; }
.metric-value { font-family: 'Space Mono', monospace; font-size: 1.6rem; font-weight: 700; color: #38bdf8; }
.metric-sub   { font-size: .75rem; color: #475569; margin-top: .2rem; }

/* Highlight badge */
.badge {
    display: inline-block;
    background: #0c4a6e;
    color: #7dd3fc;
    border-radius: 20px;
    padding: .15rem .75rem;
    font-size: .75rem;
    font-weight: 600;
    letter-spacing: .05em;
}

/* Streamlit overrides */
div[data-testid="stPlotlyChart"] { background: transparent; }
.stButton > button {
    background: linear-gradient(135deg, #0284c7, #0369a1);
    color: white; border: none; border-radius: 8px;
    padding: .55rem 1.4rem; font-weight: 600; width: 100%;
    transition: opacity .2s;
}
.stButton > button:hover { opacity: .85; }
div.stSelectbox > div { background: #1e293b; border-color: #334155; border-radius: 8px; }
</style>
""", unsafe_allow_html=True)


# ── Matplotlib theme ──────────────────────────────────────────────────────────
BG   = "#0b0f1a"
CARD = "#111827"
GRID = "#1e293b"
BLUE = "#38bdf8"
CYAN = "#06b6d4"
ROSE = "#fb7185"
GOLD = "#fbbf24"
TXT  = "#94a3b8"

def style_ax(ax):
    ax.set_facecolor(CARD)
    ax.tick_params(colors=TXT, labelsize=9)
    ax.xaxis.label.set_color(TXT)
    ax.yaxis.label.set_color(TXT)
    ax.title.set_color(BLUE)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID)

def styled_fig(nrows=1, ncols=1, figsize=(10, 5)):
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    fig.patch.set_facecolor(BG)
    if isinstance(axes, np.ndarray):
        [style_ax(a) for a in axes.flat]
    else:
        style_ax(axes)
    return fig, axes


# ── Dataset loaders ───────────────────────────────────────────────────────────
@st.cache_data
def load_dataset(name, feature=None):
    if name == "California Housing":
        raw = fetch_california_housing(as_frame=True)
        df  = raw.frame
        feat = feature or "MedInc"
        return df[[feat, "MedHouseVal"]].rename(columns={feat: "Feature", "MedHouseVal": "Target"}), raw.feature_names
    elif name == "Diabetes":
        raw = load_diabetes(as_frame=True)
        df  = raw.frame
        feat = feature or "bmi"
        return df[[feat, "target"]].rename(columns={feat: "Feature", "target": "Target"}), list(raw.feature_names)
    elif name == "Synthetic (Simple)":
        np.random.seed(42)
        n = 300
        x = np.random.uniform(0, 10, n)
        y = 3.5 * x + 8 + np.random.normal(0, 4, n)
        df = pd.DataFrame({"Feature": x, "Target": y})
        return df, ["x"]
    elif name == "Synthetic (Noisy)":
        np.random.seed(7)
        n = 400
        x = np.random.uniform(-5, 5, n)
        y = -2 * x + 1 + np.random.normal(0, 8, n)
        df = pd.DataFrame({"Feature": x, "Target": y})
        return df, ["x"]
    else:
        # Auto-MPG style
        np.random.seed(99)
        n = 250
        hp = np.random.uniform(50, 250, n)
        mpg = 50 - 0.12 * hp + np.random.normal(0, 3, n)
        df = pd.DataFrame({"Feature": hp, "Target": mpg})
        return df, ["horsepower"]


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    st.markdown("---")

    dataset_name = st.selectbox(
        "Dataset",
        ["Synthetic (Simple)", "Synthetic (Noisy)", "California Housing", "Diabetes", "Auto-MPG Style"],
    )

    feature_col = None
    if dataset_name in ["California Housing", "Diabetes"]:
        placeholder_df, feat_names = load_dataset(dataset_name)
        feature_col = st.selectbox("Feature column", feat_names)

    test_size = st.slider("Test set size (%)", 10, 40, 20, 5)
    random_state = st.slider("Random seed", 0, 99, 42)

    st.markdown("---")
    show_residuals   = st.checkbox("Show residual plot",       True)
    show_distribution = st.checkbox("Show error distribution", True)
    show_raw         = st.checkbox("Show raw data table",      False)

    st.markdown("---")
    run_btn = st.button("🚀  Train Model")


# ── Title ─────────────────────────────────────────────────────────────────────
st.markdown("# 📈 Linear Regression Lab")
st.markdown(
    '<span class="badge">PREDICTIVE MODELING</span> &nbsp; '
    '<span class="badge">SCIKIT-LEARN</span> &nbsp; '
    '<span class="badge">INTERACTIVE</span>',
    unsafe_allow_html=True,
)
st.markdown("---")

# ── Main logic ────────────────────────────────────────────────────────────────
if run_btn or True:   # auto-run on first load
    df, feat_names = load_dataset(dataset_name, feature_col)

    X = df[["Feature"]].values
    y = df["Target"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size / 100, random_state=random_state
    )

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse  = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae  = mean_absolute_error(y_test, y_pred)
    r2   = r2_score(y_test, y_pred)
    coef = model.coef_[0]
    intercept = model.intercept_

    # ── Metrics row ───────────────────────────────────────────────────────────
    st.markdown("### 📊 Model Performance")
    c1, c2, c3, c4, c5 = st.columns(5)
    metrics = [
        (c1, "R² Score",   f"{r2:.4f}",     "Variance explained"),
        (c2, "RMSE",       f"{rmse:.4f}",   "Root mean sq. error"),
        (c3, "MSE",        f"{mse:.4f}",    "Mean squared error"),
        (c4, "MAE",        f"{mae:.4f}",    "Mean absolute error"),
        (c5, "Coefficient",f"{coef:.4f}",   f"Intercept: {intercept:.4f}"),
    ]
    for col, label, value, sub in metrics:
        with col:
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="metric-label">{label}</div>'
                f'<div class="metric-value">{value}</div>'
                f'<div class="metric-sub">{sub}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Row 1: Regression line + Actual vs Predicted ──────────────────────────
    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("### 🔵 Regression Line")
        fig, ax = styled_fig(figsize=(6, 4.5))
        x_line = np.linspace(X.min(), X.max(), 200).reshape(-1, 1)
        y_line = model.predict(x_line)

        ax.scatter(X_train, y_train, color=BLUE,  alpha=.45, s=22, label="Train", zorder=2)
        ax.scatter(X_test,  y_test,  color=ROSE,  alpha=.65, s=28, label="Test",  zorder=3)
        ax.plot(x_line, y_line, color=GOLD, lw=2.5, label="Regression line", zorder=4)

        ax.set_xlabel("Feature")
        ax.set_ylabel("Target")
        ax.set_title("Data & Fitted Line", fontsize=12, fontweight='bold', pad=10)
        ax.legend(framealpha=0, labelcolor=TXT, fontsize=9)
        ax.grid(True, color=GRID, alpha=.6, lw=.6)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col_r:
        st.markdown("### 🎯 Actual vs Predicted")
        fig, ax = styled_fig(figsize=(6, 4.5))
        lo = min(y_test.min(), y_pred.min())
        hi = max(y_test.max(), y_pred.max())

        ax.scatter(y_test, y_pred, color=CYAN, alpha=.65, s=30, zorder=3)
        ax.plot([lo, hi], [lo, hi], color=GOLD, lw=2, ls="--", label="Perfect fit", zorder=4)

        ax.set_xlabel("Actual Values")
        ax.set_ylabel("Predicted Values")
        ax.set_title("Actual vs Predicted", fontsize=12, fontweight='bold', pad=10)
        ax.legend(framealpha=0, labelcolor=TXT, fontsize=9)
        ax.grid(True, color=GRID, alpha=.6, lw=.6)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

    # ── Row 2: Optional plots ─────────────────────────────────────────────────
    n_opt = sum([show_residuals, show_distribution])
    if n_opt:
        cols = st.columns(n_opt)
        idx  = 0

        if show_residuals:
            with cols[idx]:
                st.markdown("### 📉 Residuals")
                residuals = y_test - y_pred
                fig, ax = styled_fig(figsize=(6, 4))
                ax.scatter(y_pred, residuals, color=ROSE, alpha=.6, s=25, zorder=3)
                ax.axhline(0, color=GOLD, lw=1.8, ls="--")
                ax.set_xlabel("Predicted Values")
                ax.set_ylabel("Residuals")
                ax.set_title("Residual Plot", fontsize=11, fontweight='bold', pad=8)
                ax.grid(True, color=GRID, alpha=.5, lw=.6)
                fig.tight_layout()
                st.pyplot(fig)
                plt.close()
            idx += 1

        if show_distribution:
            with cols[idx]:
                st.markdown("### 📊 Error Distribution")
                residuals = y_test - y_pred
                fig, ax = styled_fig(figsize=(6, 4))
                ax.hist(residuals, bins=30, color=BLUE, alpha=.75, edgecolor=CARD)
                ax.axvline(0, color=GOLD, lw=1.8, ls="--")
                ax.set_xlabel("Residual")
                ax.set_ylabel("Frequency")
                ax.set_title("Error Distribution", fontsize=11, fontweight='bold', pad=8)
                ax.grid(True, color=GRID, alpha=.5, lw=.6)
                fig.tight_layout()
                st.pyplot(fig)
                plt.close()

    # ── Raw data ──────────────────────────────────────────────────────────────
    if show_raw:
        st.markdown("### 🗃️ Raw Data Sample")
        sample = df.sample(min(50, len(df)), random_state=random_state)
        st.dataframe(sample.style.format(precision=4), use_container_width=True)

    # ── Summary equation ──────────────────────────────────────────────────────
    st.markdown("---")
    sign = "+" if intercept >= 0 else "-"
    st.markdown(
        f"**Model equation:** &nbsp; `Target = {coef:.4f} × Feature {sign} {abs(intercept):.4f}`"
        f" &nbsp;|&nbsp; **Training samples:** {len(X_train)} &nbsp;|&nbsp; **Test samples:** {len(X_test)}"
    )
