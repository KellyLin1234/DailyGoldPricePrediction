import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

# ======================
# PAGE CONFIG
# ======================
st.set_page_config(page_title="Gold Price Prediction", layout="wide")

st.title("💰 Gold Price Prediction App (Random Forest)")
st.write("Forecast gold prices using ML (simulated long-term prediction)")

# ======================
# LOAD MODEL
# ======================
@st.cache_resource
def load_model():
    return joblib.load("models/random_forest.pkl")

model = load_model()

# ======================
# LOAD DATA
# ======================
@st.cache_data
def load_data():
    df = pd.read_csv("Gold Price.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")
    return df

df = load_data()

# ======================
# SIDEBAR
# ======================
st.sidebar.header("Settings")

years = st.sidebar.slider("Years to Predict", 1, 10, 10)

show_data = st.sidebar.checkbox("Show Data")
show_plot = st.sidebar.checkbox("Show Chart")

# ======================
# DATA PREVIEW
# ======================
if show_data:
    st.dataframe(df)

# ======================
# PLOT HISTORY
# ======================
if show_plot:
    fig, ax = plt.subplots()
    ax.plot(df["Date"], df["Price"])
    ax.set_title("Gold Price History")
    st.pyplot(fig)

# ======================
# FORECAST
# ======================
if st.button("🔮 Predict Future"):

    steps = years * 365
    predictions = []

    last_price = df["Price"].iloc[-1]
    input_data = np.array([[last_price]])

    for _ in range(steps):
        pred = model.predict(input_data)[0]
        predictions.append(pred)
        input_data = np.array([[pred]])

    future_dates = pd.date_range(
        start=df["Date"].iloc[-1],
        periods=steps + 1,
        freq="D"
    )[1:]

    forecast_df = pd.DataFrame({
        "Date": future_dates,
        "Predicted Price": predictions
    })

    st.subheader("Forecast Results")
    st.dataframe(forecast_df.head(50))

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df["Date"], df["Price"], label="Historical")
    ax.plot(forecast_df["Date"], forecast_df["Predicted Price"], label="Forecast")
    ax.legend()

    st.pyplot(fig)

    st.success("Forecast completed")
