import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# ======================
# PAGE CONFIG
# ======================
st.set_page_config(page_title="Gold Intelligence Dashboard", layout="wide")
st.title("💰 Gold Price Prediction Dashboard")

# ======================
# LOAD DATA
# ======================
df = pd.read_csv("Gold Price.csv")
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values("Date").reset_index(drop=True)

# ======================
# LOAD MODEL (BEST MODEL)
# ======================
model = joblib.load("models/gradient_boosting.pkl")

# ======================
# SIDEBAR
# ======================
st.sidebar.header("Forecast Settings")

years = st.sidebar.slider("Forecast Horizon (Years)", 1, 10, 5)
n_days = years * 365

st.sidebar.success("Active Model: Gradient Boosting (Best Model)")

# ======================
# KPI SECTION
# ======================
last_price = df['Price'].iloc[-1]

col1, col2, col3 = st.columns(3)

col1.metric("Current Price", f"{last_price:,.2f}")
col2.metric("Model", "Gradient Boosting")
col3.metric("Horizon", f"{years} Years")

# ======================
# HISTORICAL PLOT
# ======================
st.subheader("📈 Historical Trend")

fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(df['Date'], df['Price'], linewidth=2)
ax.set_title("Gold Price History")
ax.set_ylabel("Price")
st.pyplot(fig)

# ======================
# FORECAST (NO DRIFT VERSION)
# ======================

st.subheader("🔮 Forecast")

prices = df['Price'].values.tolist()

lag1 = prices[-1]
lag2 = prices[-2]
history = prices[-7:].copy()

predictions = []

alpha = 0.85  # stability factor

for _ in range(n_days):

    ma7 = np.mean(history)

    # stabilized lags (prevents explosion)
    lag1_stable = alpha * lag1 + (1 - alpha) * prices[-1]
    lag2_stable = alpha * lag2 + (1 - alpha) * prices[-2]

    X = np.array([[lag1_stable, lag2_stable, ma7]])

    pred = model.predict(X)[0]
    predictions.append(pred)

    # update lags
    lag2 = lag1
    lag1 = pred

    # IMPORTANT: keep MA window stable (prevents drift)
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
ax.set_title("Forecast (Gradient Boosting - Stable)")
ax.set_ylabel("Price")

st.pyplot(fig)

# ======================
# SUMMARY
# ======================
st.subheader("📊 Summary")

start = predictions[0]
end = predictions[-1]

change_pct = ((end - start) / start) * 100

if change_pct > 0:
    st.success(f"📈 Predicted Growth: +{change_pct:.2f}% over {years} years")
else:
    st.warning(f"📉 Predicted Drop: {change_pct:.2f}% over {years} years")
