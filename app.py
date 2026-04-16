import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

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

# Ensure LSTM_Pred exists (safe fallback if not included)
if 'LSTM_Pred' not in df.columns:
    df['LSTM_Pred'] = 0

# ======================
# LOAD MODEL
# ======================
model = joblib.load("models/hybrid_rf.pkl")

# ======================
# SIDEBAR
# ======================
st.sidebar.header("Forecast Settings")

years = st.sidebar.slider("Forecast Horizon (Years)", 1, 10, 5)
n_days = years * 365

# ======================
# KPI SECTION
# ======================
last_price = df['Price'].iloc[-1]

col1, col2, col3 = st.columns(3)
col1.metric("Current Gold Price", f"{last_price:,.2f}")
col2.metric("Model", "Hybrid RF + LSTM Feature")
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
# FORECAST ENGINE
# ======================
st.subheader("🔮 Forecast")

prices = df['Price'].tolist()
predictions = []

lag1 = prices[-1]
lag2 = prices[-2]
history = prices[-7:].copy()

for _ in range(n_days):

    # moving average feature
    ma7 = np.mean(history)

    # use last known LSTM feature (NO TensorFlow needed)
    lstm_pred = np.mean(df['LSTM_Pred'].iloc[-30:])

    # model input must match training
    X = np.array([[lag1, lag2, ma7, lstm_pred]])

    pred = model.predict(X)[0]
    predictions.append(pred)

    # update lags
    lag2 = lag1
    lag1 = pred

    # update rolling history
    history.append(df['Price'].iloc[-1])
    history = history[-7:]

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
# FORECAST PLOT
# ======================
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(forecast_df['Date'], forecast_df['Price'], color="green", linewidth=2)
ax.set_title("Gold Price Forecast (Hybrid Model)")
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
