import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# ======================
# PAGE CONFIG
# ======================
st.set_page_config(page_title="Gold Price Prediction", layout="wide")

st.title("📈 Gold Price Prediction System")
st.write("Random Forest Baseline (Stable Version)")

# ======================
# LOAD MODEL ARTIFACTS (FIXED)
# ======================
@st.cache_resource
def load_models():
    model = joblib.load("random_forest_log.pkl")
    scaler = joblib.load("scaler.pkl")
    features = joblib.load("features.pkl")
    return model, scaler, features

model, scaler, features = load_models()

# ======================
# LOAD DATA
# ======================
df = pd.read_csv("data.csv")
df = df.dropna()

# ======================
# FEATURE ENGINEERING (MUST MATCH TRAINING)
# ======================
df["Log_Return"] = np.log(df["Close"] / df["Close"].shift(1))
df = df.dropna()

# FIX: enforce correct feature order
df = df[features]

# ======================
# SCALE INPUT
# ======================
X_scaled = scaler.transform(df)

# ======================
# PREDICTION
# ======================
pred = model.predict(X_scaled)

# ======================
# RESULT DATAFRAME
# ======================
actual_prices = pd.read_csv("data.csv")["Close"].iloc[-len(pred):].values

df_result = pd.DataFrame({
    "Actual": actual_prices,
    "Predicted": pred
})

# ======================
# SIDEBAR
# ======================
st.sidebar.header("Options")
show_data = st.sidebar.checkbox("Show Data")
show_chart = st.sidebar.checkbox("Show Chart")

# ======================
# DATA VIEW
# ======================
if show_data:
    st.subheader("Dataset Preview")
    st.dataframe(df.tail())

# ======================
# CHART
# ======================
if show_chart:
    st.subheader("Actual vs Predicted")

    fig, ax = plt.subplots()
    ax.plot(df_result["Actual"].values, label="Actual")
    ax.plot(df_result["Predicted"].values, label="Predicted")
    ax.legend()

    st.pyplot(fig)

# ======================
# METRICS
# ======================
st.subheader("Model Performance")

mae = np.mean(np.abs(df_result["Actual"] - df_result["Predicted"]))
rmse = np.sqrt(np.mean((df_result["Actual"] - df_result["Predicted"]) ** 2))

col1, col2 = st.columns(2)
col1.metric("MAE", f"{mae:.2f}")
col2.metric("RMSE", f"{rmse:.2f}")
