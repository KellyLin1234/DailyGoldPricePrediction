import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# ======================
# PAGE CONFIG
# ======================
st.set_page_config(page_title="Gold Price Prediction", layout="wide")

st.title("📈 Daily Gold Price Prediction App")
st.write("Predict gold prices using Machine Learning (Random Forest as main model)")

# ======================
# LOAD MODEL + SCALER
# ======================
@st.cache_resource
def load_model():
    model = joblib.load("models/random_forest_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    return model, scaler

model, scaler = load_model()

# ======================
# LOAD DATA (optional for visualization)
# ======================
@st.cache_data
def load_data():
    df = pd.read_csv("data/Gold Price.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")
    return df

df = load_data()

# ======================
# SIDEBAR INPUT
# ======================
st.sidebar.header("🔧 Input Features")

# adjust these based on YOUR dataset features
open_price = st.sidebar.number_input("Open Price", value=2000.0)
high_price = st.sidebar.number_input("High Price", value=2010.0)
low_price = st.sidebar.number_input("Low Price", value=1990.0)
volume = st.sidebar.number_input("Volume", value=100000.0)

# feature array (IMPORTANT: must match training order)
features = np.array([[open_price, high_price, low_price, volume]])

# ======================
# PREDICTION
# ======================
if st.button("Predict Gold Price"):
    try:
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)

        st.success(f"💰 Predicted Gold Price: {prediction[0]:.2f}")

    except Exception as e:
        st.error(f"Error in prediction: {e}")

# ======================
# DATA VISUALIZATION
# ======================
st.subheader("📊 Gold Price Trend")

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(df["Date"], df["Price"], color="gold")
ax.set_title("Gold Price Over Time")
ax.set_xlabel("Date")
ax.set_ylabel("Price")
st.pyplot(fig)

# ======================
# MODEL INFO
# ======================
st.sidebar.markdown("### ℹ️ Model Info")
st.sidebar.write("Model: Random Forest Regressor (Main)")
st.sidebar.write("Scaler: MinMaxScaler")
