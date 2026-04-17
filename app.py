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
st.write("Random Forest Baseline Model (Stable Version)")

# ======================
# LOAD MODEL ARTIFACTS
# ======================
@st.cache_resource
def load_models():
    model = joblib.load("random_forest_log.pkl")joblib.dump
    model = joblib.load("gradient_boosting_log.pkl")
    model = joblib.load("models/lstm_log_model.keras") 
    model = joblib.load("hybrid_log.pkl")joblib.dump
    scaler = joblib.load("lstm_log_scaler.pkl")
    return model, scaler

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

df = df[features]

# ======================
# SCALE INPUT
# ======================
X_scaled = scaler.transform(df)

# ======================
# PREDICTION
# ======================
pred = model.predict(X_scaled)

df_result = pd.DataFrame({
    "Actual": pd.read_csv("data.csv")["Close"].iloc[-len(pred):].values,
    "Predicted": pred
})

# ======================
# SIDEBAR OPTIONS
# ======================
st.sidebar.header("Options")

show_data = st.sidebar.checkbox("Show Data")
show_chart = st.sidebar.checkbox("Show Prediction Chart")

# ======================
# DISPLAY DATA
# ======================
if show_data:
    st.subheader("Dataset Preview")
    st.dataframe(df.tail())

# ======================
# PLOT RESULTS
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
st.subheader("Model Output Summary")

mae = np.mean(np.abs(df_result["Actual"] - df_result["Predicted"]))
rmse = np.sqrt(np.mean((df_result["Actual"] - df_result["Predicted"]) ** 2))

col1, col2 = st.columns(2)
col1.metric("MAE", f"{mae:.2f}")
col2.metric("RMSE", f"{rmse:.2f}")
