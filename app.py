import streamlit as st
import numpy as np
import joblib

# ======================
# LOAD MODEL
# ======================
model = joblib.load("models/random_forest.pkl")

st.title("Gold Price Prediction App")

st.write("Enter yesterday's gold price to predict today's price")

# ======================
# USER INPUT
# ======================
lag1 = st.number_input("Yesterday Price", value=2000.0)

# ======================
# PREDICTION
# ======================
if st.button("Predict"):
    pred_diff = model.predict(np.array([[lag1]]))
    final_price = lag1 + pred_diff[0]

    st.success(f"Predicted Gold Price: {final_price:.2f}")
