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
    model = joblib.load("models/random_forest_model.pkl")  # change if needed
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
if st.button("🔮 Predict Future Prices"):

    last_value = df["Price"].values[-1]
    predictions = []

    input_value = np.array([[last_value]])

    for _ in range(future_days):
        pred = model.predict(input_value)[0]
        predictions.append(pred)
        input_value = np.array([[pred]])

    # Future dates
    future_dates = pd.date_range(
        start=df["Date"].iloc[-1],
        periods=future_days + 1,
        freq="D"
    )[1:]

    # ======================
    # RESULT TABLE
    # ======================
    result_df = pd.DataFrame({
        "Date": future_dates,
        "Predicted Price": predictions
    })

    st.subheader("📉 Future Predictions")
    st.dataframe(result_df)

    # ======================
    # PLOT RESULT
    # ======================
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    ax2.plot(df["Date"], df["Price"], label="Historical")
    ax2.plot(result_df["Date"], result_df["Predicted Price"],
             label="Prediction", linestyle="dashed")

    ax2.set_title("Gold Price Forecast")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Price")
    ax2.legend()

    st.pyplot(fig2)

    st.success("Prediction completed successfully!")
