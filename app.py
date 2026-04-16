import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# ======================
# PAGE SETUP
# ======================
st.set_page_config(page_title="Gold Price Predictor", layout="wide")
st.title("💰 Gold Price Prediction (10-Year Forecast)")

# ======================
# LOAD MODEL
# ======================
model = joblib.load("models/xgboost.pkl")

# ======================
# LOAD DATA
# ======================
df = pd.read_csv("Gold Price.csv")
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)

# ======================
# USER INPUT
# ======================
years = st.slider("Years to Predict", 1, 10, 10)
n_days = years * 365

# ======================
# INITIAL VALUES (REAL DATA ONLY)
# ======================
last_prices = df['Price'].values.tolist()

lag1 = last_prices[-1]
lag2 = last_prices[-2]

# IMPORTANT: keep MA7 grounded in REAL history only
price_history = last_prices[-7:].copy()

predictions = []

# ======================
# MULTI-STEP FORECAST (STABLE VERSION)
# ======================
for i in range(n_days):

    # MA7 stays anchored (prevents drift explosion)
    ma7 = np.mean(price_history)

    X = np.array([[lag1, lag2, ma7]])

    pred_price = model.predict(X)[0]

    # 🔥 safety clamp (prevents unrealistic explosion over long horizon)
    min_bound = min(last_prices) * 0.7
    max_bound = max(last_prices) * 1.3
    pred_price = np.clip(pred_price, min_bound, max_bound)

    predictions.append(pred_price)

    # update lags
    lag2 = lag1
    lag1 = pred_price

    # update rolling window BUT controlled
    price_history.append(pred_price)
    price_history.pop(0)

# ======================
# FUTURE DATES
# ======================
future_dates = pd.date_range(
    start=df['Date'].iloc[-1] + pd.Timedelta(days=1),
    periods=n_days
)

pred_df = pd.DataFrame({
    "Date": future_dates,
    "Predicted Price": predictions
})

# ======================
# DISPLAY
# ======================
st.subheader("📊 Forecast (Preview)")
st.dataframe(pred_df.head(30))

st.subheader("📈 Full Trend")

fig, ax = plt.subplots(figsize=(14, 6))

ax.plot(df['Date'], df['Price'], label="Historical Price")
ax.plot(pred_df['Date'], pred_df['Predicted Price'], label="10-Year Forecast")

ax.set_title("Gold Price Forecast (Stable Multi-Step)")
ax.set_xlabel("Date")
ax.set_ylabel("Price")
ax.legend()

st.pyplot(fig)

# ======================
# SUMMARY
# ======================
st.subheader("📌 Summary")

st.write("Last Actual Price:", df['Price'].iloc[-1])
st.write("Predicted Price (End of Forecast):", predictions[-1])
st.write("Total Change:", predictions[-1] - df['Price'].iloc[-1])
