import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# ======================
# PAGE CONFIG
# ======================
st.set_page_config(page_title="Gold Price Dashboard", layout="wide")

st.title("💰 Gold Price Prediction Dashboard (Stable Forecast)")

# ======================
# LOAD DATA
# ======================
df = pd.read_csv("Gold Price.csv")
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values("Date").reset_index(drop=True)

# ======================
# LOAD MODEL
# ======================
model = joblib.load("models/gradient_boosting.pkl")

# ======================
# SIDEBAR (OPTION 1 FIX)
# ======================
st.sidebar.header("Forecast Settings")

days = st.sidebar.slider("Forecast Horizon (Days)", 7, 90, 30)
n_days = days

st.sidebar.success("Model: Gradient Boosting (Stable Short-Term Forecast)")

# ======================
# KPI SECTION
# ======================
last_price = df['Price'].iloc[-1]

col1, col2, col3 = st.columns(3)

col1.metric("Current Price", f"{last_price:,.2f}")
col2.metric("Model", "Gradient Boosting")
col3.metric("Horizon", f"{days} Days")

# ======================
# HISTORICAL DATA
# ======================
st.subheader("📈 Historical Trend")

fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(df['Date'], df['Price'], linewidth=2)
ax.set_title("Gold Price History")
ax.set_xlabel("Date")
ax.set_ylabel("Price")
st.pyplot(fig)

# ======================
# FORECAST ENGINE (STABLE)
# ======================
st.subheader("🔮 Forecast")

prices = df['Price'].values.tolist()

lag1 = prices[-1]
lag2 = prices[-2]

history = prices[-7:].copy()
predictions = []

for _ in range(n_days):

    ma7 = np.mean(history)

    X = np.array([[lag1, lag2, ma7]])

    pred = model.predict(X)[0]
    predictions.append(pred)

    lag2 = lag1
    lag1 = pred

    # keep MA anchored to real data (prevents drift)
    history = prices[-7:]

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

ax.set_title("Short-Term Gold Price Forecast (Gradient Boosting)")
ax.set_xlabel("Date")
ax.set_ylabel("Price")

st.pyplot(fig)

# ======================
# SUMMARY
# ======================
st.subheader("📊 Forecast Summary")

start = predictions[0]
end = predictions[-1]

change_pct = ((end - start) / start) * 100

if change_pct > 0:
    st.success(f"📈 Expected Increase: +{change_pct:.2f}% over {days} days")
else:
    st.warning(f"📉 Expected Decrease: {change_pct:.2f}% over {days} days")
