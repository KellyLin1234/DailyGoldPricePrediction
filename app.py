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
# FEATURE ENGINEERING (MUST MATCH TRAINING)
# ======================
df['Lag1'] = df['Price'].shift(1)
df['Lag2'] = df['Price'].shift(2)
df['Lag3'] = df['Price'].shift(3)
df['MA7'] = df['Price'].rolling(7).mean()

df = df.dropna()

# ======================
# STREAMLIT UI
# ======================
st.title("💰 Gold Price Prediction App")
st.write("Predict future gold prices using Random Forest")

n_days = st.slider("Days to Predict", 1, 30, 7)

# ======================
# PREPARE LAST DATA POINT
# ======================
last_data = df.iloc[-1:].copy()

# ======================
# MULTI-STEP FORECAST
# ======================
predictions = []

current_input = last_data[['Lag1', 'Lag2', 'Lag3', 'MA7']].values.flatten()

price_history = list(df['Price'].values[-7:])  # for MA7 updates

for _ in range(n_days):
    # reshape input
    X = np.array(current_input).reshape(1, -1)

    # predict
    pred = model.predict(X)[0]
    predictions.append(pred)

    # update history
    price_history.append(pred)
    price_history.pop(0)

    # update features
    lag1 = pred
    lag2 = current_input[0]
    lag3 = current_input[1]
    ma7 = np.mean(price_history)

    current_input = [lag1, lag2, lag3, ma7]

# ======================
# DISPLAY RESULTS
# ======================
future_dates = pd.date_range(df['Date'].iloc[-1] + pd.Timedelta(days=1), periods=n_days)

pred_df = pd.DataFrame({
    "Date": future_dates,
    "Predicted Price": predictions
})

st.subheader("📈 Predictions")
st.dataframe(pred_df)

# ======================
# PLOT
# ======================
plt.figure(figsize=(10, 5))

# historical
plt.plot(df['Date'], df['Price'], label="Historical Price")

# forecast
plt.plot(pred_df['Date'], pred_df['Predicted Price'], label="Predicted Price", linestyle='dashed')

plt.legend()
plt.xlabel("Date")
plt.ylabel("Gold Price")
plt.title("Gold Price Forecast")

st.pyplot(plt)
