import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# ======================
# PAGE SETUP
# ======================
st.set_page_config(page_title="Gold Price Predictor", layout="wide")

st.title("💰 Gold Price Prediction App (Fixed Version)")
st.write("Clean forecasting with correct feature alignment + recursive prediction")

# ======================
# LOAD MODEL + SCALER
# ======================
@st.cache_resource
def load_assets():
    model = joblib.load("models/hybrid_rf.pkl")
    scaler = joblib.load("models/scaler.pkl")
    return model, scaler

model, scaler, features = load_assets()

# ======================
# LOAD DATA
# ======================
@st.cache_data
def load_data():
    df = pd.read_csv("data/Gold Price.csv")  # adjust path if needed
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")
    df = df.reset_index(drop=True)
    return df

df = load_data()

# ======================
# CREATE LAG FEATURES
# ======================
def create_features(data):
    data = data.copy()

    data["lag1"] = data["Close"].shift(1)
    data["lag2"] = data["Close"].shift(2)
    data["lag3"] = data["Close"].shift(3)

    data = data.dropna()
    return data

df_feat = create_features(df)

# ======================
# SIDEBAR SETTINGS
# ======================
st.sidebar.header("Forecast Settings")
steps = st.sidebar.slider("Years to Predict", 1, 10, 5)

# ======================
# PREDICTION INPUT
# ======================
def get_last_input(df_feat):
    last_row = df_feat[features].iloc[-1].values
    return last_row

# ======================
# RECURSIVE FORECAST
# ======================
def forecast(model, scaler, last_input, steps):
    predictions = []
    current = last_input.copy()

    for _ in range(steps):
        X = current.reshape(1, -1)
        X_scaled = scaler.transform(X)

        pred = model.predict(X_scaled)[0]
        predictions.append(pred)

        # shift lag values
        current = np.roll(current, 1)
        current[0] = pred  # update lag1 with new prediction

    return predictions

# ======================
# RUN FORECAST
# ======================
last_input = get_last_input(df_feat)
preds = forecast(model, scaler, last_input, steps)

# create future dates
last_date = df_feat["Date"].iloc[-1]
future_dates = pd.date_range(last_date, periods=steps + 1, freq="Y")[1:]

# ======================
# DISPLAY RESULTS
# ======================
st.subheader("📊 Forecast Results")

forecast_df = pd.DataFrame({
    "Date": future_dates,
    "Predicted Price": preds
})

st.dataframe(forecast_df)

# ======================
# PLOT
# ======================
fig, ax = plt.subplots()

ax.plot(df_feat["Date"].tail(100), df_feat["Close"].tail(100), label="Historical")
ax.plot(forecast_df["Date"], forecast_df["Predicted Price"], label="Forecast")

ax.set_title("Gold Price Forecast")
ax.legend()

st.pyplot(fig)

# ======================
# METRICS (optional simple check)
# ======================
st.subheader("📉 Model Info")
st.write("Features used:", features)
