import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# ======================
# PAGE CONFIG
# ======================
st.set_page_config(page_title="Gold Price Prediction", layout="wide")

st.title("📈 Gold Price Prediction App")
st.write("Best Model: Random Forest Regressor")

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

# (optional - keep if you want comparison)
try:
    xgb_model = joblib.load("models/xgboost.pkl")
except:
    xgb_model = None

# ======================
# SIDEBAR MENU
# ======================
menu = st.sidebar.radio("Menu", ["Data", "Visualization", "Prediction"])

# ======================
# DATA VIEW
# ======================
if menu == "Data":
    st.subheader("📊 Dataset")
    st.write(data.tail(20))

# ======================
# VISUALIZATION
# ======================
elif menu == "Visualization":
    st.subheader("📉 Gold Price Trend")

    fig, ax = plt.subplots()
    ax.plot(data["Date"], data["Price"])
    ax.set_xlabel("Date")
    ax.set_ylabel("Gold Price")
    st.pyplot(fig)

# ======================
# PREDICTION
# ======================
elif menu == "Prediction":
    st.subheader("🔮 Next Day Gold Price Prediction")

    # Features used in training
    features = ["SPX", "SLV", "USDX"]

    latest_data = data[features].values[-1].reshape(1, -1)

    # ======================
    # RANDOM FOREST (MAIN MODEL)
    # ======================
    rf_pred = rf_model.predict(latest_data)[0]

    st.success(f"🌳 Random Forest Prediction: {rf_pred:.2f}")

    # ======================
    # OPTIONAL XGBOOST
    # ======================
    if xgb_model is not None:
        xgb_pred = xgb_model.predict(latest_data)[0]
        st.info(f"⚡ XGBoost Prediction: {xgb_pred:.2f}")

        # Hybrid (optional comparison)
        hybrid = (rf_pred + xgb_pred) / 2
        st.warning(f"🔗 Hybrid Prediction: {hybrid:.2f}")

    # ======================
    # SIMPLE BAR CHART
    # ======================
    st.subheader("📊 Model Comparison")

    models = ["Random Forest"]
    values = [rf_pred]

    if xgb_model is not None:
        models.append("XGBoost")
        values.append(xgb_pred)

        models.append("Hybrid")
        values.append(hybrid)

    fig, ax = plt.subplots()
    ax.bar(models, values)
    ax.set_ylabel("Price")
    st.pyplot(fig)
