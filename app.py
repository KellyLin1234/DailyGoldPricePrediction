import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# ======================
# PAGE SETUP
# ======================
st.set_page_config(page_title="Gold Price Predictor", layout="wide")
st.title("💰 Gold Price Prediction (Stable Multi-Step Forecast)")

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
# INITIAL HISTORY (REAL DATA ONLY)
# ======================
history = df['Price'].values.tolist()

predictions = []

# ======================
# STABLE FORECAST LOOP
# ======================
for _ in range(n_days):

    # Always compute features from REAL recent window (NOT growing predictions)
    lag1 = history[-1]
    lag2 = history[-2]
    ma7 = np.mean(history[-7:])

    X = np.array([[lag1, lag2, ma7]])

    pred_price = model.predict(X)[0]

    predictions.append(pred_price)

    # Update history for rolling features
    history.append(pred_price)

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
st.subheader("📊 Forecast (First 30 Days)")
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
# SUMMARY
# ======================
st.subheader("📈 Summary")

st.write("Last Actual Price:", df['Price'].iloc[-1])
st.write("Final Forecast Price:", predictions[-1])
st.write("Total Change:", predictions[-1] - df['Price'].iloc[-1])
