import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn import metrics

# --- UI & Styling ---
st.set_page_config(page_title="Ultimate Regression Suite", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #0e1117; color: #fafafa; }
    .stNumberInput input { background-color: #161b22; color: #58a6ff; }
    div[data-testid="column"] { 
        padding: 1rem; 
        border-radius: 10px; 
        background: #161b22; 
        border: 1px solid #30363d;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    st.title("⚙️ Engine Room")
    file = st.file_uploader("Upload CSV Data", type=['csv'])
    if file:
        df = pd.read_csv(file).select_dtypes(include=[np.number]).dropna()
        st.success("Data Ready")
        
        st.divider()
        model_choice = st.radio("Optimization Strategy", ["Ridge (L2)", "Lasso (L1)"])
        max_deg = st.slider("Complexity (Max Degree)", 1, 4, 2)
        
if not file:
    st.info("👋 Welcome! Please upload a CSV to build your predictive model.")
    st.stop()

# --- Feature Engineering ---
st.header("🎯 Model Targeting")
col_t, col_f = st.columns([1, 2])
with col_t:
    target = st.selectbox("Target Variable", df.columns, index=len(df.columns)-1)
with col_f:
    features = st.multiselect("Predictor Variables", [c for c in df.columns if c != target], default=df.columns[0])

if not features:
    st.warning("Select features to continue.")
    st.stop()

# --- Optimization Process ---
X = df[features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define Search
base_model = Ridge() if "Ridge" in model_choice else Lasso(max_iter=5000)
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('poly', PolynomialFeatures()),
    ('regressor', base_model)
])

param_grid = {
    'poly__degree': list(range(1, max_deg + 1)),
    'regressor__alpha': [0.01, 0.1, 1, 10, 100]
}

grid = GridSearchCV(pipe, param_grid, cv=5, scoring='r2')
grid.fit(X_train, y_train)
best_model = grid.best_estimator_

# --- Results & Visuals ---
st.divider()
res1, res2, res3 = st.columns(3)
res1.metric("Optimized R²", f"{metrics.r2_score(y_test, best_model.predict(X_test)):.4f}")
res2.metric("Best Degree", grid.best_params_['poly__degree'])
res3.metric("Best λ", grid.best_params_['regressor__alpha'])

# Tabs for Analysis
tab_perf, tab_resid, tab_pred = st.tabs(["📊 Performance", "📉 Error Analysis", "🔮 LIVE PREDICTOR"])

with tab_perf:
    y_pred = best_model.predict(X_test)
    fig_p = px.scatter(x=y_test, y=y_pred, labels={'x':'Actual','y':'Predicted'}, title="Model Accuracy")
    fig_p.add_shape(type="line", x0=y.min(), y0=y.min(), x1=y.max(), y1=y.max(), line=dict(color="red", dash="dot"))
    st.plotly_chart(fig_p, use_container_width=True)

with tab_resid:
    residuals = y_test - y_pred
    fig_dist = ff.create_distplot([residuals], ['Residuals'], colors=['#58a6ff'], bin_size=.5)
    fig_dist.update_layout(title_text='Residual Distribution (Goal: Normal Curve)')
    st.plotly_chart(fig_dist, use_container_width=True)
    st.caption("If this curve is centered at 0 and bell-shaped, your model is statistically unbiased.")

with tab_pred:
    st.subheader("Input New Data for Prediction")
    input_data = {}
    
    # Create dynamic inputs based on selected features
    input_cols = st.columns(len(features))
    for i, col in enumerate(input_cols):
        with col:
            val = st.number_input(f"{features[i]}", value=float(df[features[i]].mean()))
            input_data[features[i]] = [val]
    
    if st.button("Generate Prediction"):
        new_df = pd.DataFrame(input_data)
        prediction = best_model.predict(new_df)[0]
        
        st.markdown(f"""
            <div style="padding:20px; border-radius:10px; background-color:#238636; text-align:center;">
                <h2 style="color:white; margin:0;">Predicted {target}</h2>
                <h1 style="color:white; margin:0;">{prediction:.2f}</h1>
            </div>
        """, unsafe_allow_html=True)
