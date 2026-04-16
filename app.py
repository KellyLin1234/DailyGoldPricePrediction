import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# ======================
# PAGE SETUP
# ======================
st.set_page_config(page_title="Gold Price Predictor", layout="wide")

st.title("💰 Gold Price Prediction (Multi-Step Forecast)")

# ======================
# LOAD MODEL
# ======================
model = joblib.load("models/random_forest.pkl")

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
# MULTI-STEP FORECAST
# ======================

predictions = []

last_prices = df['Price'].values.tolist()

lag1 = last_prices[-1]
lag2 = last_prices[-2]

price_history = last_prices[-7:].copy()

for _ in range(n_days):

    # MA7 feature
    ma7 = np.mean(price_history)

    # Feature vector (MUST match training)
    X = np.array([[lag1, lag2, ma7]])

    # Predict next price
    pred_price = model.predict(X)[0]

    predictions.append(pred_price)

    # Update lag values
    lag2 = lag1
    lag1 = pred_price

    # Update rolling window
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
# DISPLAY TABLE
# ======================
st.subheader("📊 Forecast Data")
st.dataframe(pred_df.head(30))

# ======================
# PLOT
# ======================
fig, ax = plt.subplots(figsize=(12, 5))

ax.plot(df['Date'], df['Price'], label="Historical Price")
ax.plot(pred_df['Date'], pred_df['Predicted Price'], label="Forecast")

ax.set_title("Gold Price Forecast")
ax.set_xlabel("Date")
ax.set_ylabel("Price")
ax.legend()

st.pyplot(fig)

# ======================
# SUMMARY METRICS
# ======================
st.subheader("📈 Forecast Summary")

st.write("Last Actual Price:", df['Price'].iloc[-1])
st.write("Predicted End Price:", predictions[-1])
st.write("Total Change:", predictions[-1] - df['Price'].iloc[-1])
