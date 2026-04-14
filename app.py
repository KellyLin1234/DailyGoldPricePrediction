import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from utils.preprocessing import clean_data

st.title("📊 Gold Price Prediction System")

# Load data
df = pd.read_csv("data/Gold Price.csv")
df = clean_data(df)

st.subheader("Dataset")
st.write(df.head())

# Model selection
model_choice = st.selectbox(
    "Choose Model",
    ["Random Forest", "Gradient Boosting", "XGBoost", "LSTM", "Hybrid"]
)

# Load models
rf = joblib.load("models/random_forest.pkl")
gb = joblib.load("models/gb_model.pkl")
xgb = joblib.load("models/xgb_model.pkl")

lstm = tf.keras.models.load_model("models/lstm_model.h5")
hybrid_rf = joblib.load("models/hybrid_rf.pkl")
hybrid_lstm = tf.keras.models.load_model("models/hybrid_lstm.h5")

# Input
st.subheader("Input Features")

open_p = st.number_input("Open")
high_p = st.number_input("High")
low_p = st.number_input("Low")
volume = st.number_input("Volume")
chg = st.number_input("Chg%")

date = st.date_input("Date")

if st.button("Predict"):

    date_val = pd.Timestamp(date).toordinal()

    features = np.array([[date_val, open_p, high_p, low_p, volume, chg]])

    if model_choice == "Random Forest":
        pred = rf.predict([[open_p]])[0]  # based on your lag model

    elif model_choice == "Gradient Boosting":
        pred = gb.predict(features)[0]

    elif model_choice == "XGBoost":
        pred = xgb.predict(features)[0]

    elif model_choice == "LSTM":
        st.warning("LSTM uses sequence input — demo prediction only")
        pred = "Use full sequence input (handled in backend)"

    elif model_choice == "Hybrid":
        st.warning("Hybrid uses LSTM + RF pipeline")
        pred = "Run full pipeline in backend"

    st.success(f"Predicted Price: {pred}")
