import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

# ======================
# PAGE CONFIG
# ======================
st.set_page_config(page_title="Gold Price Forecast", layout="wide")

st.title("💰 Gold Price Prediction App (XGBoost)")
st.write("10-Year Forecast using Machine Learning (XGBoost Model)")

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
st.sidebar.header("Settings")

years = st.sidebar.slider("Years to Predict", 1, 10, 10)
show_data = st.sidebar.checkbox("Show Data")
show_plot = st.sidebar.checkbox("Show Chart")

# ======================
# DATA DISPLAY
# ======================
if show_data:
    st.subheader("📊 Dataset")
    st.dataframe(df)

if show_plot:
    st.subheader("📈 Gold Price History")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df["Date"], df["Price"])
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.set_title("Gold Price Trend")

    st.pyplot(fig)

# ======================
# FEATURE ENGINEERING
# MUST MATCH TRAINING FEATURES
# ======================
df["Lag1"] = df["Price"].shift(1)
df["Lag2"] = df["Price"].shift(2)
df["Lag3"] = df["Price"].shift(3)
df["Lag7"] = df["Price"].shift(7)
df["MA7"] = df["Price"].rolling(7).mean()

df = df.dropna()

# ======================
# PREDICTION
# ======================
if st.button("🔮 Predict 10-Year Forecast"):

    steps = years * 365
    predictions = []

    last_values = df["Price"].iloc[-7:].values  # last 7 days window

    progress = st.progress(0)

    for i in range(steps):

        lag1 = last_values[-1]
        lag2 = last_values[-2]
        lag3 = last_values[-3]
        lag7 = last_values[-7]
        ma7 = np.mean(last_values[-7:])

        X_input = np.array([[lag1, lag2, lag3, lag7, ma7]])

        pred = model.predict(X_input)[0]
        predictions.append(pred)

        # update rolling window
        last_values = np.append(last_values, pred)
        last_values = last_values[-7:]

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
    # RESULTS
    # ======================
    st.subheader("📊 10-Year Forecast Result")
    st.dataframe(forecast_df.tail(50))

    # ======================
    # PLOT
    # ======================
    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(df["Date"], df["Price"], label="Historical", linewidth=2)
    ax.plot(forecast_df["Date"], forecast_df["Predicted Price"],
            label="10-Year Forecast", linestyle="dashed")

    ax.set_title("Gold Price 10-Year Forecast (XGBoost)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()

    st.pyplot(fig)

    st.success("10-year forecast completed successfully!")
