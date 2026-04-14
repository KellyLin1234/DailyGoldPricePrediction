import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

# ======================
# CONFIG
# ======================
st.set_page_config(page_title="Gold Price Forecast", layout="wide")

st.title("💰 Gold Price Forecast Dashboard")
st.caption("Model: XGBoost (Lag1 Feature)")

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
# FEATURE
# ======================
df["Lag1"] = df["Price"].shift(1)
df = df.dropna()

# ======================
# SIDEBAR
# ======================
st.sidebar.header("Settings")
years = st.sidebar.slider("Years to Predict", 1, 10, 10)

# ======================
# METRICS
# ======================
col1, col2, col3 = st.columns(3)

col1.metric("Latest Price", f"${df['Price'].iloc[-1]:,.2f}")
col2.metric("Average Price", f"${df['Price'].mean():,.2f}")
col3.metric("Max Price", f"${df['Price'].max():,.2f}")

st.divider()

# ======================
# FORECAST
# ======================
if st.button("🔮 Generate Forecast"):

    steps = years * 365
    predictions = []

    last_price = df["Price"].iloc[-1]

    progress = st.progress(0)

    for i in range(steps):

        X_input = pd.DataFrame({"Lag1": [last_price]})
        pred = model.predict(X_input)[0]

        predictions.append(pred)
        last_price = pred

        if i % 100 == 0:
            progress.progress(i / steps)

    # ======================
    # FUTURE DATES
    # ======================
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
    # RESULTS
    # ======================
    st.subheader("📊 Forecast Results")
    st.dataframe(forecast_df.tail(30))

    # ======================
    # CHART
    # ======================
    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(df["Date"], df["Price"], label="Historical")
    ax.plot(forecast_df["Date"], forecast_df["Predicted Price"],
            label="Forecast", linestyle="dashed")

    ax.legend()
    ax.set_title("Gold Price Forecast (XGBoost Lag1)")

    st.pyplot(fig)

    # ======================
    # DOWNLOAD
    # ======================
    csv = forecast_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "Download Forecast",
        csv,
        "forecast.csv",
        "text/csv"
    )
