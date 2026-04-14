import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# ======================
# PAGE CONFIG
# ======================
st.set_page_config(page_title="Gold Price Prediction", layout="wide")

st.title("📈 Gold Price Prediction App")
st.write("Compare ML + Deep Learning models and predict gold prices")

# ======================
# LOAD DATA
# ======================
@st.cache_data
def load_data():
    df = pd.read_csv("data/Gold Price.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    return df

data = load_data()

# ======================
# LOAD MODELS
# ======================
rf_model = joblib.load("models/random_forest.pkl")
xgb_model = joblib.load("models/xgboost.pkl")

lstm_model = load_model("models/lstm_model.h5")

# ======================
# SIDEBAR
# ======================
st.sidebar.header("📌 Navigation")
option = st.sidebar.radio("Go to:", ["Data", "Charts", "Prediction"])

# ======================
# DATA VIEW
# ======================
if option == "Data":
    st.subheader("📊 Dataset Preview")
    st.write(data.tail(20))

# ======================
# CHARTS
# ======================
elif option == "Charts":
    st.subheader("📉 Gold Price Trend")

    fig, ax = plt.subplots()
    ax.plot(data["Date"], data["Price"])
    ax.set_xlabel("Date")
    ax.set_ylabel("Gold Price")
    st.pyplot(fig)

# ======================
# PREDICTION
# ======================
elif option == "Prediction":
    st.subheader("🔮 Next Day Price Prediction")

    # ======================
    # INPUT FEATURES (latest row used)
    # ======================
    features = ["SPX", "SLV", "USDX"]

    latest_input = data[features].values[-1].reshape(1, -1)

    # ======================
    # ML PREDICTIONS
    # ======================
    rf_pred = rf_model.predict(latest_input)[0]
    xgb_pred = xgb_model.predict(latest_input)[0]

    # ======================
    # LSTM (simple fallback)
    # NOTE: assumes you already prepared correct LSTM input in training
    # ======================
    try:
        lstm_input = latest_input.reshape((1, latest_input.shape[1], 1))
        lstm_pred = lstm_model.predict(lstm_input)[0][0]
    except:
        lstm_pred = np.nan

    # ======================
    # HYBRID MODEL
    # ======================
    hybrid_pred = (rf_pred + xgb_pred) / 2

    # ======================
    # DISPLAY RESULTS
    # ======================
    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Random Forest", f"{rf_pred:.2f}")
    col2.metric("XGBoost", f"{xgb_pred:.2f}")
    col3.metric("LSTM", f"{lstm_pred:.2f}" if not np.isnan(lstm_pred) else "N/A")
    col4.metric("Hybrid", f"{hybrid_pred:.2f}")

    # ======================
    # CHART COMPARISON
    # ======================
    st.subheader("📊 Model Comparison")

    fig, ax = plt.subplots()
    models = ["RF", "XGB", "LSTM", "Hybrid"]
    values = [
        rf_pred,
        xgb_pred,
        lstm_pred if not np.isnan(lstm_pred) else 0,
        hybrid_pred
    ]

    ax.bar(models, values)
    st.pyplot(fig)
