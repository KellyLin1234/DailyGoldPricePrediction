import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# ======================
# PAGE CONFIG
# ======================
st.set_page_config(page_title="Gold Price Prediction", layout="wide")

# ======================
# LOAD MODEL + SCALER
# ======================
model = joblib.load("models/random_forest.pkl")
scaler = joblib.load("models/scaler.pkl")

# ======================
# LOAD DATA
# ======================
df = pd.read_csv("data/Gold Price.csv")
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)

# ======================
# FEATURE ENGINEERING (MUST MATCH TRAINING)
# ======================
df['Lag1'] = df['Price'].shift(1)
df['Lag2'] = df['Price'].shift(2)
df['Lag3'] = df['Price'].shift(3)
df['MA7'] = df['Price'].rolling(7).mean()

df = df.dropna()

# ======================
# UI
# ======================
st.title("💰 Gold Price Prediction App")
st.markdown("Predict future gold prices using Machine Learning (Random Forest)")

n_days = st.slider("Select number of days to predict", 1, 30, 7)

# ======================
# GET LAST DATA
# ======================
last_row = df.iloc[-1]

lag1 = last_row['Price']
lag2 = df.iloc[-2]['Price']
lag3 = df.iloc[-3]['Price']

price_history = list(df['Price'].values[-7:])

# ======================
# MULTI-STEP FORECAST
# ======================
predictions = []

for _ in range(n_days):

    ma7 = np.mean(price_history)

    X = np.array([[lag1, lag2, lag3, ma7]])
    X_scaled = scaler.transform(X)

    pred = model.predict(X_scaled)[0]
    predictions.append(pred)

    # update lags
    lag3 = lag2
    lag2 = lag1
    lag1 = pred

    # update MA7
    price_history.append(pred)
    price_history.pop(0)

# ======================
# CREATE RESULT DF
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
st.subheader("📊 Prediction Results")
st.dataframe(pred_df, use_container_width=True)

# ======================
# PLOT
# ======================
fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(df['Date'], df['Price'], label="Historical Price")
ax.plot(pred_df['Date'], pred_df['Predicted Price'],
        label="Predicted Price", linestyle='dashed')

ax.set_title("Gold Price Forecast")
ax.set_xlabel("Date")
ax.set_ylabel("Price")
ax.legend()

st.pyplot(fig)

# ======================
# METRICS (OPTIONAL BUT GOOD FOR MARKS)
# ======================
st.subheader("📌 Latest Info")
st.write(f"Last Known Price: {df['Price'].iloc[-1]:.2f}")
st.write(f"Next Day Prediction: {predictions[0]:.2f}")
