import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# ======================
# LOAD MODEL
# ======================
model = joblib.load("models/random_forest.pkl")
scaler = joblib.load("models/scaler.pkl")

# ======================
# LOAD DATA
# ======================
df = pd.read_csv("data/Gold Price.csv")
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)

st.title("💰 Gold Price Prediction (10-Year Forecast)")

years = st.slider("Years to Predict", 1, 10, 10)
n_days = years * 365

# ======================
# INITIAL VALUES
# ======================
prices = df['Price'].values.tolist()

predictions = []

# ======================
# ROLLING FORECAST
# ======================
for _ in range(n_days):

    lag1 = prices[-1]
    lag2 = prices[-2]
    lag3 = prices[-3]
    ma7 = np.mean(prices[-7:])

    X = np.array([[lag1, lag2, lag3, ma7]])
    X_scaled = scaler.transform(X)

    pred = model.predict(X_scaled)[0]

    predictions.append(pred)
    prices.append(pred)

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
st.subheader("📊 10-Year Forecast")
st.dataframe(pred_df.head(30))  # show first 30 days only

# ======================
# PLOT (sample view)
# ======================
fig, ax = plt.subplots(figsize=(12, 5))

ax.plot(df['Date'], df['Price'], label="History")
ax.plot(pred_df['Date'], pred_df['Predicted Price'], label="Forecast")

ax.set_title("10-Year Gold Price Prediction")
ax.legend()

st.pyplot(fig)
