import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

# ======================
# PAGE CONFIG
# ======================
st.set_page_config(page_title="Gold Price Forecast", layout="wide")

st.title("💰 Gold Price Prediction App (XGBoost - Lag1 Only)")
st.write("10-Year Forecast using simplified ML model")

# ======================
# LOAD MODEL
# ======================
@st.cache_resource
def load_model():
    return joblib.load("models/xgboost.pkl")

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
years = st.sidebar.slider("Years to Predict", 1, 10, 10)
show_data = st.sidebar.checkbox("Show Data")
show_chart = st.sidebar.checkbox("Show Chart")

# ======================
# DATA VIEW
# ======================
if show_data:
    st.dataframe(df)

if show_chart:
    fig, ax = plt.subplots()
    ax.plot(df["Date"], df["Price"])
    ax.set_title("Gold Price History")
    st.pyplot(fig)

# ======================
# FORECAST
# ======================
if st.button("🔮 Predict 10-Year Forecast"):

    steps = years * 365
    predictions = []

    # last known price
    last_price = df["Price"].iloc[-1]

    progress = st.progress(0)

    for i in range(steps):

        # ONLY LAG1 INPUT (FIXED)
        X_input = pd.DataFrame({"Lag1": [last_price]})

        pred = model.predict(X_input)[0]
        predictions.append(pred)

        # recursive update
        last_price = pred

        if i % 100 == 0:
            progress.progress(i / steps)

    # future dates
    future_dates = pd.date_range(
        start=df["Date"].iloc[-1],
        periods=steps + 1,
        freq="D"
    )[1:]

    forecast_df = pd.DataFrame({
        "Date": future_dates,
        "Predicted Price": predictions
    })

    # ======================
    # RESULT
    # ======================
    st.subheader("📊 10-Year Forecast Result")
    st.dataframe(forecast_df.tail(50))

    # ======================
    # PLOT
    # ======================
    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(df["Date"], df["Price"], label="Historical")
    ax.plot(forecast_df["Date"], forecast_df["Predicted Price"],
            label="10-Year Forecast", linestyle="dashed")

    ax.set_title("Gold Price 10-Year Forecast (XGBoost - Lag1)")
    ax.legend()

    st.pyplot(fig)

    st.success("Forecast completed successfully!")
