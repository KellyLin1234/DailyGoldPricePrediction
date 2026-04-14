import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

# ======================
# PAGE CONFIG
# ======================
st.set_page_config(
    page_title="Gold Price Forecast",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ======================
# TITLE
# ======================
st.title("💰 Gold Price Forecast Dashboard")
st.markdown("### XGBoost Model • 10-Year Prediction")

st.divider()

# ======================
# LOAD MODEL
# ======================
@st.cache_resource
def load_model():
    return joblib.load("models/xgboost.pkl")

model = load_model()

# ======================
# LOAD DATA
# ======================
@st.cache_data
def load_data():
    df = pd.read_csv("Gold Price.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")
    return df

df = load_data()

# ======================
# SIDEBAR
# ======================
st.sidebar.header("⚙️ Controls")

years = st.sidebar.slider("Forecast Horizon (Years)", 1, 10, 10)
show_data = st.sidebar.toggle("Show Dataset")
show_chart = st.sidebar.toggle("Show Historical Chart")

# ======================
# METRICS (TOP CARDS)
# ======================
last_price = df["Price"].iloc[-1]
avg_price = df["Price"].mean()
max_price = df["Price"].max()

col1, col2, col3 = st.columns(3)

col1.metric("Latest Gold Price", f"${last_price:,.2f}")
col2.metric("Average Price", f"${avg_price:,.2f}")
col3.metric("Peak Price", f"${max_price:,.2f}")

st.divider()

# ======================
# DATA TABLE
# ======================
if show_data:
    st.subheader("📊 Dataset Preview")
    st.dataframe(df, use_container_width=True)

# ======================
# HISTORICAL CHART
# ======================
if show_chart:
    st.subheader("📈 Historical Gold Price Trend")

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df["Date"], df["Price"], linewidth=2)

    ax.set_title("Gold Price Over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")

    st.pyplot(fig)

# ======================
# FORECAST BUTTON
# ======================
st.divider()

if st.button("🚀 Generate 10-Year Forecast", use_container_width=True):

    steps = years * 365
    predictions = []

    last_price = df["Price"].iloc[-1]

    progress = st.progress(0)
    status = st.empty()

    for i in range(steps):

        X_input = pd.DataFrame({"Lag1": [last_price]})

        pred = model.predict(X_input)[0]
        predictions.append(pred)

        last_price = pred

        # progress update
        if i % 100 == 0:
            progress.progress(i / steps)
            status.info(f"Generating forecast... {i}/{steps} days")

    status.success("Forecast completed!")

    # ======================
    # FUTURE DATES
    # ======================
    future_dates = pd.date_range(
        start=df["Date"].iloc[-1],
        periods=steps + 1,
        freq="D"
    )[1:]

    forecast_df = pd.DataFrame({
        "Date": future_dates,
        "Predicted Price": predictions
    })

    # ======================
    # RESULTS HEADER
    # ======================
    st.subheader("📊 Forecast Results")

    st.dataframe(forecast_df.tail(30), use_container_width=True)

    # ======================
    # FORECAST CHART
    # ======================
    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(df["Date"], df["Price"], label="Historical", linewidth=2)
    ax.plot(
        forecast_df["Date"],
        forecast_df["Predicted Price"],
        label="10-Year Forecast",
        linestyle="dashed"
    )

    ax.set_title("Gold Price Forecast (XGBoost - Lag1 Model)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()

    st.pyplot(fig)

    # ======================
    # DOWNLOAD BUTTON
    # ======================
    csv = forecast_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="📥 Download Forecast CSV",
        data=csv,
        file_name="gold_price_forecast_10y.csv",
        mime="text/csv"
    )

    st.success("Done! Forecast ready 🎉")
