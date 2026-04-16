import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# ======================
# LOAD MODEL + SCALER
# ======================
model = joblib.load("models/rf_model.pkl")
scaler = joblib.load("models/scaler.pkl")

# ======================
# LOAD DATA
# ======================
df = pd.read_csv("data/Gold Price.csv")
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)

# ======================
# FEATURE (OLD MODEL STYLE: ONLY LAG1)
# ======================
df['Lag1'] = df['Price'].shift(1)
df = df.dropna()

st.title("💰 Gold Price Prediction App (Legacy Model)")

n_days = st.slider("Days to Predict", 1, 30, 7)

# ======================
# START FROM LAST KNOWN VALUE
# ======================
lag1 = df['Price'].iloc[-1]

predictions = []

for _ in range(n_days):

    # ONLY 1 FEATURE (IMPORTANT)
    X = np.array([[lag1]])

    # scale
    X_scaled = scaler.transform(X)

    # predict
    pred = model.predict(X_scaled)[0]
    predictions.append(pred)

    # update lag
    lag1 = pred

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
# OUTPUT
# ======================
st.subheader("📊 Predictions")
st.dataframe(pred_df)

# ======================
# PLOT
# ======================
fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(df['Date'], df['Price'], label="Historical")
ax.plot(pred_df['Date'], pred_df['Predicted Price'], label="Forecast", linestyle='dashed')

ax.set_title("Gold Price Forecast (Legacy Model)")
ax.set_xlabel("Date")
ax.set_ylabel("Price")
ax.legend()

st.pyplot(fig)
