import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.preprocessing import MinMaxScaler

# ======================
# PAGE CONFIG
# ======================
st.set_page_config(page_title="Gold Price Prediction", layout="wide")

st.title("💰 Gold Price Prediction App (Random Forest)")
st.write("Predict future gold prices using Machine Learning (Random Forest Model)")

# ======================
# LOAD MODEL
# ======================
@st.cache_resource
def load_model():
    model = joblib.load("models/random_forest.pkl")
    return model

model = load_model()

# ======================
# LOAD DATA
# ======================
@st.cache_data
def load_data():
    df = pd.read_csv("Gold Price.csv")  # change path if needed
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")
    return df

df = load_data()

# ======================
# SIDEBAR
# ======================
st.sidebar.header("📌 Settings")

future_days = st.sidebar.slider("Days to Predict", 1, 60, 7)

show_data = st.sidebar.checkbox("Show Raw Data")
show_plot = st.sidebar.checkbox("Show Price Chart")

# ======================
# DATA PREVIEW
# ======================
if show_data:
    st.subheader("📊 Dataset")
    st.dataframe(df)

# ======================
# BASIC PLOT
# ======================
if show_plot:
    st.subheader("📈 Gold Price History")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df["Date"], df["Price"], label="Gold Price")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()

    st.pyplot(fig)

# ======================
# FEATURE ENGINEERING (simple lag example)
# ======================
df["Lag1"] = df["Price"].shift(1)
df = df.dropna()

X = df[["Lag1"]]
y = df["Price"]

# ======================
# PREDICTION
# ======================
if st.button("🔮 Predict Next 10 Years"):

    years = 10
    steps = years * 365

    predictions = []

    # start with last known price
    last_price = df["Price"].values[-1]

    input_data = np.array([[last_price]])

    for _ in range(steps):
        pred = model.predict(input_data)[0]
        predictions.append(pred)

        # feed prediction back into model (recursive)
        input_data = np.array([[pred]])

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

    st.subheader("📊 10-Year Gold Price Forecast")
    st.dataframe(forecast_df.head(50))  # show first 50 rows only

    # plot
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df["Date"], df["Price"], label="Historical")
    ax.plot(forecast_df["Date"], forecast_df["Predicted Price"], label="10-Year Forecast")

    ax.set_title("Gold Price 10-Year Prediction (Simulated)")
    ax.legend()

    st.pyplot(fig)

    st.success("10-year forecast generated!")
