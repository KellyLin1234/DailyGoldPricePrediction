import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ======================
# 1. LOAD MODELS
# ======================
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
features = joblib.load("features.pkl")

# ======================
# 2. LOAD DATA
# ======================
df = pd.read_csv("data/Gold Price.csv")
df = df.sort_values("Date")

# ======================
# 3. APP UI
# ======================
st.title("💰 Gold Price Prediction App")

years = st.slider("Years to Predict", 1, 10, 1)

# ======================
# 4. PREP INPUT
# ======================
df["Close_lag1"] = df["Close"].shift(1)
df["Close_lag2"] = df["Close"].shift(2)
df = df.dropna()

X = df[features]

# IMPORTANT: match training scaler
X_scaled = scaler.transform(X)

# ======================
# 5. FORECAST FUNCTION
# ======================
def forecast(last_row, steps):
    preds = []

    row = last_row.copy()

    for _ in range(steps):
        X_input = scaler.transform([row])
        pred = model.predict(X_input)[0]
        preds.append(pred)

        # update lag features correctly
        row["Close_lag2"] = row["Close_lag1"]
        row["Close_lag1"] = pred
        row["Close"] = pred

    return preds

# ======================
# 6. RUN FORECAST
# ======================
if st.button("Predict"):
    last_row = df[features].iloc[-1].copy()

    steps = years * 365
    preds = forecast(last_row, steps)

    st.write("Prediction completed")

    st.line_chart(preds)
