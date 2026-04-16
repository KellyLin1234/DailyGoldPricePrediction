import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt

# ======================
# PAGE CONFIG
# ======================
st.set_page_config(page_title="Gold Price Predictor", layout="wide")

st.title("💰 Gold Price Prediction App (Stable Version)")
st.write("Fixed version: no missing file dependencies + correct forecasting")

# ======================
# PATH SAFETY (IMPORTANT)
# ======================
BASE_DIR = os.path.dirname(__file__)

MODEL_PATH = os.path.join(BASE_DIR, "models/model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models/scaler.pkl")

# ======================
# LOAD MODEL + SCALER
# ======================
@st.cache_resource
def load_assets():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler

model, scaler = load_assets()

# ======================
# FIXED FEATURES (NO FILE NEEDED)
# ======================
FEATURES = ["lag1", "lag2", "lag3"]

# ======================
# LOAD DATA
# ======================
@st.cache_data
def load_data():
    df = pd.read_csv(os.path.join(BASE_DIR, "data/Gold Price.csv"))
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    return df

df = load_data()

# ======================
# CREATE FEATURES
# ======================
def create_features(data):
    data = data.copy()
    data["lag1"] = data["Close"].shift(1)
    data["lag2"] = data["Close"].shift(2)
    data["lag3"] = data["Close"].shift(3)
    return data.dropna()

df_feat = create_features(df)

# ======================
# SIDEBAR
# ======================
st
