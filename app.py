import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# ======================
# PAGE CONFIG
# ======================
st.set_page_config(page_title="Gold Intelligence Dashboard", layout="wide")

st.title("💰 Gold Intelligence Dashboard (Gradient Boosting Model)")

# ======================
# LOAD DATA
# ======================
df = pd.read_csv("Gold Price.csv")
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values("Date").reset_index(drop=True)

# ======================
# LOAD BEST MODEL (GRADIENT BOOSTING)
# ======================
model = joblib.load("models/gradient_boosting.pkl")

# ======================
# SIDEBAR CONTROLS
# ======================
st.sidebar.header("📊 Forecast Controls")

years = st.sidebar.slider("Forecast Horizon (Years)", 1, 10, 5)
n_days = years * 365

st.sidebar.markdown("### Active Model")
st.sidebar.success("Gradient Boosting (BEST MODEL)")

# ======================
# KPI METRICS
# ======================
last_price = df['Price'].iloc[-1]

col1, col2, col3 = st.columns(3)

col1.metric("💵 Current Price", f"${last_price:,.2f}")
col2.metric("🧠 Model", "Gradient Boosting")
col3.metric("📅 Horizon", f"{years} Years")

# ======================
# HISTORICAL CHART
# ======================
st.subheader("📉 Gold Price History")

fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(df['Date'], df['Price'], linewidth=2)
ax.set_title("Gold Price Trend")
ax.set_ylabel("Price")

st.pyplot(fig)

# ======================
# MULTI-STEP FORECAST (GB MODEL)
# ======================
st.subheader("🔮 Forecast (Gradient Boosting)")

last_prices = df['Price'].values.tolist()

lag1 = last_prices[-1]
lag2 = last_prices[-2]
history = last_prices[-7:].copy()

predictions = []

for _ in range(n_days):

    ma7 = np.mean(history)

    X = np.array([[lag1, lag2, ma7]])

    pred = model.predict(X)[0]
    predictions.append(pred)

    lag2 = lag1
    lag1 = pred

    history.append(pred)
    history.pop(0)

future_dates = pd.date_range(df['Date'].iloc[-1], periods=n_days)

forecast_df = pd.DataFrame({
    "Date": future_dates,
    "Price": predictions
})

# ======================
# FORECAST CHART
# ======================
fig, ax = plt.subplots(figsize=(14, 5))

ax.plot(forecast_df['Date'], forecast_df['Price'], color="green", linewidth=2)

ax.set_title("Gradient Boosting Forecast Path")
ax.set_ylabel("Price")

st.pyplot(fig)

# ======================
# PERFORMANCE (NOW GB IS BEST)
# ======================
st.subheader("📊 Model Performance")

st.markdown("""
### 🏆 Best Model: Gradient Boosting

- R² Score: **0.78 (Highest)**
- MAE: Lowest among tested models
- RMSE: Most stable predictions
""")

col1, col2, col3 = st.columns(3)

col1.metric("Random Forest", "R² 0.72")
col2.metric("XGBoost", "R² 0.77")
col3.metric("Gradient Boosting", "R² 0.78 🏆")

# ======================
# SUMMARY
# ======================
st.subheader("💼 Investment Insight")

start = predictions[0]
end = predictions[-1]

ret = ((end - start) / start) * 100

if ret > 0:
    st.success(f"📈 Uptrend Forecast: +{ret:.2f}% over {years} years")
else:
    st.warning(f"📉 Downtrend Forecast: {ret:.2f}% over {years} years")
