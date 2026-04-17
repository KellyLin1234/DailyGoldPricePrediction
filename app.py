import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# ======================
# PAGE CONFIG
# ======================
st.set_page_config(page_title="Gold Intelligence Dashboard", layout="wide")
st.title("💰 Gold Price Prediction Dashboard (Hybrid Model)")

# ======================
# LOAD DATA
# ======================
df = pd.read_csv("Gold Price.csv")
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values("Date").reset_index(drop=True)

# ======================
# LOAD MODELS
# ======================
hybrid_model = joblib.load("models/hybrid_rf.pkl")
lstm_model = load_model("models/lstm_model.h5")

# If you used scaler during training (VERY IMPORTANT)
scaler = joblib.load("models/scaler.pkl")

# ======================
# SIDEBAR
# ======================
st.sidebar.header("Forecast Settings")

years = st.sidebar.slider("Forecast Horizon (Years)", 1, 10, 5)
n_days = years * 365

# ======================
# KPI
# ======================
last_price = df['Price'].iloc[-1]

col1, col2, col3 = st.columns(3)
col1.metric("Current Gold Price", f"{last_price:,.2f}")
col2.metric("Model", "Hybrid RF + LSTM")
col3.metric("Forecast Horizon", f"{years} Years")

# ======================
# HISTORICAL PLOT
# ======================
st.subheader("📈 Historical Gold Price")

fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(df['Date'], df['Price'], linewidth=2)
ax.set_title("Gold Price History")
ax.set_xlabel("Date")
ax.set_ylabel("Price")
st.pyplot(fig)

# ======================
# FEATURE PREP
# ======================
prices = df['Price'].values.tolist()

def create_lstm_input(prices, window=60):
    arr = np.array(prices[-window:])
    return arr.reshape(1, window, 1)

# ======================
# FORECAST ENGINE
# ======================
st.subheader("🔮 Forecast")

predictions = []

lag1 = prices[-1]
lag2 = prices[-2]

history = prices[-7:].copy()

for _ in range(n_days):

    ma7 = np.mean(history)

    # ===== LSTM FEATURE =====
    lstm_input = create_lstm_input(prices)
    lstm_pred = lstm_model.predict(lstm_input, verbose=0)[0][0]

    # inverse transform if needed
    lstm_pred = scaler.inverse_transform([[lstm_pred]])[0][0]

    # ===== HYBRID INPUT =====
    X = np.array([[lag1, lag2, ma7, lstm_pred]])

    pred = hybrid_model.predict(X)[0]
    predictions.append(pred)

    # update lags
    lag2 = lag1
    lag1 = pred

    # update history
    history.append(pred)
    history = history[-7:]
    prices.append(pred)

# ======================
# FUTURE DATES
# ======================
future_dates = pd.date_range(
    start=df['Date'].iloc[-1] + pd.Timedelta(days=1),
    periods=n_days
)

forecast_df = pd.DataFrame({
    "Date": future_dates,
    "Price": predictions
})

# ======================
# PLOT FORECAST
# ======================
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(forecast_df['Date'], forecast_df['Price'], color="green", linewidth=2)
ax.set_title("Hybrid Model Forecast")
ax.set_xlabel("Date")
ax.set_ylabel("Price")
st.pyplot(fig)

# ======================
# SUMMARY
# ======================
st.subheader("📊 Forecast Summary")

start_price = predictions[0]
end_price = predictions[-1]

change_pct = ((end_price - start_price) / start_price) * 100

if change_pct > 0:
    st.success(f"📈 Expected Growth: +{change_pct:.2f}% over {years} years")
else:
    st.warning(f"📉 Expected Decline: {change_pct:.2f}% over {years} years")
