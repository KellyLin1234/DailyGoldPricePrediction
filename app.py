import streamlit as st
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
import matplotlib.pyplot as plt
from collections import deque

# ======================
# PAGE CONFIG
# ======================
st.set_page_config(page_title="Gold Price Forecast", layout="wide")

# ======================
# LOAD DATA
# ======================
df = pd.read_csv("Gold Price.csv")  # change if needed
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')

df['Price'] = df['Price'].ffill()
df['Price_Log'] = np.log(df['Price'])

# ======================
# LOAD MODELS
# ======================
rf = joblib.load("models/random_forest_log.pkl")
gb = joblib.load("models/gradient_boosting_log.pkl")
features = joblib.load("models/features.pkl")

lstm = tf.keras.models.load_model("models/lstm_log_model.keras")
scaler = joblib.load("models/lstm_log_scaler.pkl")

hybrid = joblib.load("models/hybrid_log.pkl")

# ======================
# INIT SIDEBAR
# ======================
st.sidebar.title("Forecast Settings")
years = st.sidebar.slider("Forecast Years", 1, 10, 10)
days = years * 365

model_choice = st.sidebar.selectbox(
    "Select Model",
    ["Random Forest", "Gradient Boosting", "LSTM", "Hybrid", "All"]
)

# ======================
# FEATURE ENGINEERING FUNCTION
# ======================
def create_features(df):
    df = df.copy()
    df['Lag1'] = df['Price_Log'].shift(1)
    df['Lag2'] = df['Price_Log'].shift(2)
    df['Lag3'] = df['Price_Log'].shift(3)
    df['Lag5'] = df['Price_Log'].shift(5)
    df['Lag10'] = df['Price_Log'].shift(10)

    df['MA7'] = df['Price_Log'].rolling(7).mean()
    df['MA14'] = df['Price_Log'].rolling(14).mean()
    df['MA30'] = df['Price_Log'].rolling(30).mean()

    df['Volatility7'] = df['Price_Log'].rolling(7).std()
    df['Volatility14'] = df['Price_Log'].rolling(14).std()

    df['Momentum'] = df['Price_Log'] - df['Price_Log'].shift(5)

    return df.dropna()

# ======================
# INITIAL DATA
# ======================
df_feat = create_features(df)
latest = df_feat.iloc[-1]

history = df_feat['Price_Log'].tolist()

# ======================
# FORECAST FUNCTIONS
# ======================

def rf_forecast(history, steps):
    result = []
    hist = history.copy()

    for _ in range(steps):
        df_tmp = pd.DataFrame({"Price_Log": hist})

        df_tmp = create_features(df_tmp)
        x = df_tmp[features].iloc[-1:].values

        pred = rf.predict(x)[0]
        next_val = df_tmp['Lag1'].iloc[-1] + pred

        hist.append(next_val)
        result.append(np.exp(next_val))

    return result


def gb_forecast(history, steps):
    result = []
    hist = history.copy()

    for _ in range(steps):
        df_tmp = pd.DataFrame({"Price_Log": hist})
        df_tmp = create_features(df_tmp)

        x = df_tmp[features].iloc[-1:].values
        pred = gb.predict(x)[0]

        next_val = df_tmp['Lag1'].iloc[-1] + pred
        hist.append(next_val)
        result.append(np.exp(next_val))

    return result


def lstm_forecast(history, steps):
    result = []
    window = deque(history[-60:], maxlen=60)

    for _ in range(steps):
        x = np.array(window).reshape(1, 60, 1)
        pred = lstm.predict(x, verbose=0)[0][0]

        pred = scaler.inverse_transform([[pred]])[0][0]
        next_val = window[-1] + pred

        window.append(next_val)
        result.append(np.exp(next_val))

    return result


def hybrid_forecast(history, steps):
    rf_res = rf_forecast(history, steps)
    gb_res = gb_forecast(history, steps)
    lstm_res = lstm_forecast(history, steps)

    return [
        (r + g + l) / 3
        for r, g, l in zip(rf_res, gb_res, lstm_res)
    ]

# ======================
# RUN FORECAST
# ======================
if st.sidebar.button("Run Forecast"):

    if model_choice == "Random Forest":
        forecast = rf_forecast(history, days)

    elif model_choice == "Gradient Boosting":
        forecast = gb_forecast(history, days)

    elif model_choice == "LSTM":
        forecast = lstm_forecast(history, days)

    elif model_choice == "Hybrid":
        forecast = hybrid_forecast(history, days)

    else:
        rf_f = rf_forecast(history, days)
        gb_f = gb_forecast(history, days)
        lstm_f = lstm_forecast(history, days)

        forecast = (np.array(rf_f) + np.array(gb_f) + np.array(lstm_f)) / 3

    # ======================
    # PLOT
    # ======================
    st.subheader("10-Year Gold Price Forecast")

    fig, ax = plt.subplots()
    ax.plot(range(len(forecast)), forecast)
    ax.set_title("Forecasted Gold Price")
    ax.set_xlabel("Days")
    ax.set_ylabel("Price")

    st.pyplot(fig)

    st.success("Forecast completed successfully!")

# ======================
# SHOW DATA
# ======================
st.subheader("Latest Data")
st.write(df.tail())
