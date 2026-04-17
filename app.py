import streamlit as st
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model

# ======================
# PAGE CONFIG
# ======================
st.set_page_config(page_title="Gold Price Forecast", layout="wide")

st.title("💰 Gold Price Prediction System (RF + GB + LSTM + Hybrid)")

# ======================
# LOAD DATA
# ======================
df = pd.read_csv("data/Gold Price.csv")
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)

df['Price'] = df['Price'].ffill()
df['Price_Log'] = np.log(df['Price'])
df['Log_Return'] = df['Price_Log'].diff()
df = df.dropna().reset_index(drop=True)

# ======================
# LOAD MODELS
# ======================
rf = joblib.load("models/random_forest_log.pkl")
gb = joblib.load("models/gradient_boosting_log.pkl")
lstm = load_model("models/lstm_log_model.keras")
scaler = joblib.load("models/lstm_log_scaler.pkl")

# ======================
# FEATURE ENGINEERING
# ======================
def make_features(data):
    data = data.copy()
    data['Lag1'] = data['Price_Log'].shift(1)
    data['Lag2'] = data['Price_Log'].shift(2)
    data['Lag3'] = data['Price_Log'].shift(3)
    data['Lag5'] = data['Price_Log'].shift(5)
    data['Lag10'] = data['Price_Log'].shift(10)

    data['MA7'] = data['Price_Log'].rolling(7).mean()
    data['MA14'] = data['Price_Log'].rolling(14).mean()
    data['MA30'] = data['Price_Log'].rolling(30).mean()

    data['Volatility7'] = data['Price_Log'].rolling(7).std()
    data['Volatility14'] = data['Price_Log'].rolling(14).std()

    data['Momentum'] = data['Price_Log'] - data['Price_Log'].shift(5)

    return data.dropna()

# ======================
# WINDOW FUNCTION (LSTM)
# ======================
def create_window(data, window=60):
    X = []
    for i in range(window, len(data)):
        X.append(data[i-window:i, 0])
    return np.array(X)

# ======================
# FORECAST FUNCTION
# ======================
def forecast(model_name, steps=3650):

    temp_df = df.copy()

    rf_features = [
        'Lag1','Lag2','Lag3','Lag5','Lag10',
        'MA7','MA14','MA30',
        'Volatility7','Volatility14',
        'Momentum'
    ]

    prices = temp_df['Price_Log'].values.tolist()

    lstm_window = 60
    lstm_buffer = scaler.transform(temp_df[['Log_Return']].values)

    for _ in range(steps):

        # update feature frame
        temp = pd.DataFrame({"Price_Log": prices})
        temp = make_features(temp)

        latest = temp.iloc[-1]

        rf_input = np.array([latest[rf_features].values])
        rf_pred = rf.predict(rf_input)[0]

        gb_pred = gb.predict(rf_input)[0]

        # LSTM window
        lstm_window_data = lstm_buffer[-lstm_window:]
        lstm_input = lstm_window_data.reshape(1, lstm_window, 1)

        lstm_pred_scaled = lstm.predict(lstm_input, verbose=0)
        lstm_pred = scaler.inverse_transform(lstm_pred_scaled)[0][0]

        # HYBRID (clean averaging)
        hybrid_pred = (rf_pred + gb_pred + lstm_pred) / 3

        if model_name == "Random Forest":
            pred = rf_pred
        elif model_name == "Gradient Boosting":
            pred = gb_pred
        elif model_name == "LSTM":
            pred = lstm_pred
        else:
            pred = hybrid_pred

        new_log_price = prices[-1] + pred
        prices.append(new_log_price)

        lstm_buffer = np.vstack([
            lstm_buffer,
            scaler.transform([[pred]])
        ])

    return np.exp(prices)

# ======================
# SIDEBAR
# ======================
model_choice = st.sidebar.selectbox(
    "Select Model",
    ["Random Forest", "Gradient Boosting", "LSTM", "Hybrid"]
)

years = st.sidebar.slider("Forecast Years", 1, 10, 5)
steps = years * 365

# ======================
# RUN FORECAST
# ======================
if st.button("Run Forecast"):

    result = forecast(model_choice, steps)

    st.subheader(f"📈 {model_choice} Forecast ({years} Years)")

    fig, ax = plt.subplots()
    ax.plot(result)
    ax.set_title("Gold Price Forecast")
    ax.set_xlabel("Days")
    ax.set_ylabel("Price")
    st.pyplot(fig)

    st.subheader("📊 Final Value")
    st.write(f"Predicted Price after {years} years: **{result[-1]:.2f}**")

# ======================
# MODEL COMPARISON TABLE (STATIC)
# ======================
st.subheader("📊 Model Comparison (Your Results)")

comparison = pd.DataFrame({
    "Model": ["Random Forest", "Gradient Boosting", "LSTM", "Hybrid"],
    "MAE": [602, 493, 556, 337],
    "RMSE": [946, 784, 874, 511],
    "R2": [0.9979, 0.9985, 0.9982, 0.9992],
    "MAPE": [0.69, 0.56, 0.63, 0.40]
})

st.dataframe(comparison)

st.success("System ready: clean hybrid + stable 10-year forecast")
