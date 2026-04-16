import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from datetime import timedelta

# ======================
# 1. PAGE SETUP & LOADING
# ======================
st.set_page_config(page_title="Gold Price Predictor", layout="wide")

@st.cache_resource
def load_assets():
    # Loading the Random Forest model
    # Note: Ensure "models/random_forest.pkl" exists!
    model = joblib.load("models/random_forest.pkl")
    return model

try:
    rf_model = load_assets()
except:
    st.error("Model file not found. Please run your training script first.")
    st.stop()

# ======================
# 2. DATA PREPARATION
# ======================
st.title("💰 Gold Price Forecasting")
st.markdown("This app uses a **Random Forest Regressor** to predict gold price trends.")

# Load the dataset to get the most recent values
@st.cache_data
def get_data():
    df = pd.read_csv('Gold Price.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    return df

df = get_data()

# Sidebar inputs
st.sidebar.header("Forecast Settings")
n_days = st.sidebar.slider("Days to Predict", 1, 30, 7)

# ======================
# 3. MULTI-STEP FORECAST LOGIC
# ======================
# We need Lag1, Lag2, and MA7 to match your training:
# X_train = train[['Lag1', 'Lag2', 'MA7']]

# Get the most recent data points
last_prices = df['Price'].tolist()

predictions = []
current_history = last_prices.copy()

for i in range(n_days):
    # Prepare features for the current step
    lag1 = current_history[-1]
    lag2 = current_history[-2]
    ma7  = np.mean(current_history[-7:])
    
    # Create feature array (Matching the 3 features from training)
    features = np.array([[lag1, lag2, ma7]])
    
    # Predict the DIFFERENCE (as your training used y_train = train['Price_Diff'])
    pred_diff = rf_model.predict(features)[0]
    
    # Calculate actual price: Price = Last_Price + Predicted_Diff
    next_price = lag1 + pred_diff
    
    predictions.append(next_price)
    current_history.append(next_price)

# ======================
# 4. VISUALIZATION
# ======================
future_dates = [df['Date'].iloc[-1] + timedelta(days=x) for x in range(1, n_days + 1)]
pred_df = pd.DataFrame({'Date': future_dates, 'Predicted Price': predictions})

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Forecasted Prices")
    st.dataframe(pred_df.style.format({"Predicted Price": "{:.2f}"}))

with col2:
    st.subheader("Price Chart")
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Show last 30 days of history + forecast
    recent_hist = df.tail(30)
    ax.plot(recent_hist['Date'], recent_hist['Price'], label="Actual Price", color="blue")
    ax.plot(pred_df['Date'], pred_df['Predicted Price'], label="Forecast", color="orange", linestyle="--")
    
    plt.xticks(rotation=45)
    ax.legend()
    st.pyplot(fig)

# ======================
# 5. METRICS
# ======================
st.divider()
m1, m2, m3 = st.columns(3)
m1.metric("Last Closing Price", f"${last_prices[-1]:,.2f}")
m2.metric("Next Day Forecast", f"${predictions[0]:,.2f}", f"{predictions[0]-last_prices[-1]:+.2f}")
m3.metric("Avg Forecasted Price", f"${np.mean(predictions):,.2f}")
