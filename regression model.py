# --------------------------------------------------------------
# app.py  –  Simple Linear Regression demo with Streamlit
# --------------------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# --------------------------------------------------------------
# 0. Page configuration
# --------------------------------------------------------------
st.set_page_config(
    page_title="Simple Linear Regression Demo",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------
# 1. Title & description
# --------------------------------------------------------------
st.title("📊 Simple Linear Regression with Streamlit")
st.markdown(
    """
    This app demonstrates a **single‑feature linear regression** on the classic **Diabetes** data set
    (continuous target: disease progression).  

    * Choose a predictor variable.  
    * Adjust the train‑test split.  
    * See the fitted regression line, performance metrics, and a comparison of actual vs. predicted values.
    """
)

# --------------------------------------------------------------
# 2. Load & cache the dataset
# --------------------------------------------------------------
@st.cache_data
def load_data():
    """Load the diabetes dataset and return a tidy DataFrame."""
    diabetes = load_diabetes()
    # The data are already standardized; keep them that way for simplicity.
    df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
    df["target"] = diabetes.target
    return df, diabetes.feature_names      # <-- unchanged

df, feature_names = load_data()

# --------------------------------------------------------------
# 3. Sidebar – user controls
# --------------------------------------------------------------
st.sidebar.header("⚙️ Settings")

# 3.1 Feature selector (simple regression → single column)
default_feature = "bmi"
if default_feature in feature_names:
    default_idx = feature_names.index(default_feature)
else:
    default_idx = 0               # fall back to the first feature

selected_feature = st.sidebar.selectbox(
    "Select predictor (X) → simple linear regression",
    options=feature_names,
    index=default_idx,           # <-- fixed line
)

# 3.2 Train‑test split ratio
test_size = st.sidebar.slider(
    "Test set proportion",
    min_value=0.1,
    max_value=0.5,
    value=0.2,
    step=0.05,
    help="Fraction of the data that will be held out for testing."
)

# 3.3 Random state for reproducibility
random_state = st.sidebar.number_input(
    "Random seed",
    min_value=0,
    max_value=9999,
    value=42,
    step=1,
)

# --------------------------------------------------------------
# 4. Prepare X and y
# --------------------------------------------------------------
X = df[[selected_feature]].values  # shape (n_samples, 1)
y = df["target"].values

# --------------------------------------------------------------
# 5. Train‑test split
# --------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=random_state
)

# --------------------------------------------------------------
# 6. Model training
# --------------------------------------------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# --------------------------------------------------------------
# 7. Predictions
# --------------------------------------------------------------
y_train_pred = model.predict(X_train)
y_test_pred  = model.predict(X_test)

# --------------------------------------------------------------
# 8. Evaluation metrics
# --------------------------------------------------------------
def compute_metrics(y_true, y_pred, dataset_name):
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    st.metric(label=f"{dataset_name} MSE", value=f"{mse:,.2f}")
    st.metric(label=f"{dataset_name} R²", value=f"{r2:.3f}")

st.subheader("🔎 Model performance")
col1, col2 = st.columns(2)
with col1:
    st.write("**Training set**")
    compute_metrics(y_train, y_train_pred, "Train")
with col2:
    st.write("**Test set**")
    compute_metrics(y_test, y_test_pred, "Test")

# --------------------------------------------------------------
# 9. Visualisation – Regression line + points
# --------------------------------------------------------------
st.subheader("📈 Regression line & data points")
fig, ax = plt.subplots(figsize=(8, 5))

# Plot training points
ax.scatter(
    X_train,
    y_train,
    color="steelblue",
    alpha=0.6,
    label="Train data",
    edgecolor="k",
)

# Plot test points
ax.scatter(
    X_test,
    y_test,
    color="orange",
    alpha=0.6,
    label="Test data",
    edgecolor="k",
)

# Create a dense X range for the line
x_line = np.linspace(X.min() - 0.1, X.max() + 0.1, 200).reshape(-1, 1)
y_line = model.predict(x_line)
ax.plot(x_line, y_line, color="darkred", linewidth=2, label="Fitted line")

ax.set_xlabel(selected_feature.upper())
ax.set_ylabel("Target (disease progression)")
ax.set_title(f"Linear regression on '{selected_feature}'")
ax.legend()
ax.grid(True, linestyle="--", alpha=0.5)

st.pyplot(fig)

# --------------------------------------------------------------
# 10. Visualisation – Actual vs. Predicted (test set)
# --------------------------------------------------------------
st.subheader("🔎 Actual vs. Predicted (test set)")
fig2, ax2 = plt.subplots(figsize=(8, 5))

ax2.scatter(
    y_test,
    y_test_pred,
    color="mediumpurple",
    alpha=0.7,
    edgecolor="k",
    label="Test observations",
)

# 45‑degree reference line (perfect predictions)
lims = [
    np.min([y_test.min(), y_test_pred.min()]) - 5,
    np.max([y_test.max(), y_test_pred.max()]) + 5,
]
ax2.plot(lims, lims, "k--", linewidth=1, label="Ideal (y = ŷ)")

ax2.set_xlabel("Actual target")
ax2.set_ylabel("Predicted target")
ax2.set_title("Actual vs. Predicted values")
ax2.set_xlim(lims)
ax2.set_ylim(lims)
ax2.legend()
ax2.grid(True, linestyle="--", alpha=0.5)

st.pyplot(fig2)

# --------------------------------------------------------------
# 11. Download predictions
# --------------------------------------------------------------
st.subheader("💾 Download predictions")
pred_df = pd.DataFrame({
    "X_" + selected_feature: X_test.ravel(),
    "actual_target": y_test,
    "predicted_target": y_test_pred,
})
csv = pred_df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="Download test predictions as CSV",
    data=csv,
    file_name="test_predictions.csv",
    mime="text/csv",
)

# --------------------------------------------------------------
# 12. Optional: show raw data (expandable)
# --------------------------------------------------------------
with st.expander("📂 Show first rows of the dataset"):
    st.dataframe(df.head())

st.caption(
    """
    *Data source*: `sklearn.datasets.load_diabetes`.  
    The features are already standardized (zero mean, unit variance).  
    The target is a quantitative measure of disease progression one year after baseline.
    """
)
