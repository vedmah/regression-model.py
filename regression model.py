import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# --- Page Config ---
st.set_page_config(page_title="Linear Regression Studio", layout="wide")

# --- Custom CSS for Styling ---
st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    h1 {
        color: #1E3A8A;
        font-family: 'Helvetica Neue', sans-serif;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Header ---
st.title("📈 Linear Regression Dashboard")
st.write("Upload a CSV to train a simple linear model and visualize results.")

# --- Sidebar: Data Upload ---
with st.sidebar:
    st.header("1. Data Setup")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.success("File uploaded!")
        
        st.header("2. Model Settings")
        test_size = st.slider("Test Set Size (%)", 10, 50, 20) / 100
        random_state = st.number_input("Random Seed", value=42)

# --- Main Logic ---
if uploaded_file:
    # Column Selection
    cols = df.columns.tolist()
    col1, col2 = st.columns(2)
    
    with col1:
        feature = st.selectbox("Select Independent Variable (X)", cols)
    with col2:
        target = st.selectbox("Select Target Variable (y)", cols)

    # Prepare Data
    X = df[[feature]].values
    y = df[target].values
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Model Training
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # --- Results Display ---
    st.divider()
    m1, m2, m3 = st.columns(3)
    m1.metric("R-Squared", f"{r2:.4f}")
    m2.metric("MSE", f"{mse:.2f}")
    m3.metric("Slope (Coef)", f"{model.coef_[0]:.2f}")

    # --- Visualizations ---
    tab1, tab2 = st.tabs(["Regression Line", "Actual vs. Predicted"])

    with tab1:
        # Generate points for the regression line
        x_range = np.linspace(X.min(), X.max(), 100)
        y_range = model.predict(x_range.reshape(-1, 1))

        fig_line = px.scatter(df, x=feature, y=target, opacity=0.6, title="Best Fit Line")
        fig_line.add_traces(go.Scatter(x=x_range, y=y_range, name="Regression Line", line=dict(color='red', width=3)))
        st.plotly_chart(fig_line, use_container_width=True)

    with tab2:
        results_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
        fig_pred = px.scatter(results_df, x='Actual', y='Predicted', 
                              hover_data=['Actual', 'Predicted'],
                              title="Prediction Accuracy (Ideal: 45° Line)")
        fig_pred.add_shape(type="line", x0=y_test.min(), y0=y_test.min(), x1=y_test.max(), y1=y_test.max(),
                           line=dict(color="Green", dash="dot"))
        st.plotly_chart(fig_pred, use_container_width=True)

else:
    st.info("Please upload a CSV file in the sidebar to get started. Tip: Ensure your columns are numeric!")
