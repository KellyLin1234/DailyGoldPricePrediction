import streamlit as st
import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from collections import deque

# ======================
# PAGE CONFIG
# ======================
st.set_page_config(page_title="Gold Price Forecast (PyTorch)", layout="wide")

# ======================
# LOAD DATA
# ======================
df = pd.read_csv("Gold Price.csv")

df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')

df['Price'] = df['Price'].ffill()
df['Price_Log'] = np.log(df['Price'])

# ======================
# LOAD SKLEARN MODELS
# ======================
rf = joblib.load("models/random_forest_log.pkl")
gb = joblib.load("models/gradient_boosting_log.pkl")
features = joblib.load("models/features.pkl")

# ======================
# DEVICE
# ======================
device = torch.device("cpu")

# ======================
# PYTORCH LSTM MODEL (MUST MATCH TRAINING ARCHITECTURE)
# ======================
class LSTMModel(nn.Module):
    def __init__(self):
        super(LSTMModel, self).__init__()
        self.lstm1 = nn.LSTM(input_size=1, hidden_size=64, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=64, hidden_size=32, batch_first=True)
        self.fc = nn.Linear(32, 1)

    def forward(self, x):
        out, _ = self.lstm1(x)
        out, _ = self.lstm2(out)
        out = self.fc(out[:, -1, :])
        return out

# ======================
# LOAD LSTM (PYTORCH .pkl)
# ======================
lstm = LSTMModel()
lstm.load_state_dict(torch.load("models/lstm_model.pkl", map_location=device))
lstm.eval()

# ======================
# SIDEBAR
# ======================
st.sidebar.title("Forecast Settings")

years = st.sidebar.slider("Forecast Years", 1, 10, 10)
days = years * 365

model_choice = st.sidebar.selectbox(
    "Select Model",
    ["Random Forest", "Gradient Boosting", "LSTM", "Hybrid", "All"]
)

# ======================
# FEATURE ENGINEERING
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

df_feat = create_features(df)
history = df_feat['Price_Log'].tolist()

# ======================
# RF FORECAST
# ======================
def rf_forecast(history, steps):
    hist = history.copy()
    result = []

    for _ in range(steps):
        tmp = pd.DataFrame({"Price_Log": hist})
        tmp = create_features(tmp)

        x = tmp[features].iloc[-1:].values
        pred = rf.predict(x)[0]

        next_val = tmp['Lag1'].iloc[-1] + pred
        hist.append(next_val)

        result.append(np.exp(next_val))

    return result

# ======================
# GB FORECAST
# ======================
def gb_forecast(history, steps):
    hist = history.copy()
    result = []

    for _ in range(steps):
        tmp = pd.DataFrame({"Price_Log": hist})
        tmp = create_features(tmp)

        x = tmp[features].iloc[-1:].values
        pred = gb.predict(x)[0]

        next_val = tmp['Lag1'].iloc[-1] + pred
        hist.append(next_val)

        result.append(np.exp(next_val))

    return result

# ======================
# LSTM FORECAST (PYTORCH .pkl)
# ======================
def lstm_forecast(history, steps):
    hist = deque(history[-60:], maxlen=60)
    result = []

    for _ in range(steps):
        x = np.array(hist, dtype=np.float32).reshape(1, 60, 1)
        x = torch.tensor(x)

        with torch.no_grad():
            pred = lstm(x).item()

        next_val = hist[-1] + pred
        hist.append(next_val)

        result.append(np.exp(next_val))

    return result

# ======================
# HYBRID FORECAST
# ======================
def hybrid_forecast(history, steps):
    rf_res = rf_forecast(history, steps)
    gb_res = gb_forecast(history, steps)
    lstm_res = lstm_forecast(history, steps)

    return [(r + g + l) / 3 for r, g, l in zip(rf_res, gb_res, lstm_res)]

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
    ax.plot(forecast)
    ax.set_title("Gold Price Prediction (10 Years)")
    ax.set_xlabel("Days")
    ax.set_ylabel("Price")

    st.pyplot(fig)

    # ======================
    # TABLE
    # ======================
    st.subheader("Forecast Data")
    st.write(pd.DataFrame({"Forecast Price": forecast}))

    st.success("Forecast completed successfully!")

# ======================
# DATA PREVIEW
# ======================
st.subheader("Latest Data")
st.write(df.tail())
