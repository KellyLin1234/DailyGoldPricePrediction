import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from PIL import Image

# ======================
# PAGE CONFIG
# ======================
st.set_page_config(page_title="Gold Forecast (Stable Version)", layout="wide")

def load_img(path):
    return Image.open(path)

# ======================
# LOAD DATA
# ======================
df = pd.read_csv("Gold Price.csv")

df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')

df['Price'] = df['Price'].ffill()
df['Price_Log'] = np.log(df['Price'])

# ======================
# LOAD MODELS
# ======================
rf = joblib.load("models/random_forest_log.pkl")
gb = joblib.load("models/gradient_boosting_log.pkl")
features = joblib.load("models/features.pkl")

# ======================
# SIDEBAR
# ======================
st.sidebar.title("Forecast Settings")

years = st.sidebar.slider("Forecast Years", 1, 10, 10)
days = years * 365

model_choice = st.sidebar.selectbox(
    "Select Model",
    ["Random Forest", "Gradient Boosting", "Hybrid", "All"]
)

# ======================
# SAFE FEATURE ENGINEERING
# ======================
def create_features(df):
    df = df.copy()

    df['Lag1'] = df['Price_Log'].shift(1)
    df['Lag2'] = df['Price_Log'].shift(2)
    df['Lag3'] = df['Price_Log'].shift(3)
    df['Lag5'] = df['Price_Log'].shift(5)
    df['Lag10'] = df['Price_Log'].shift(10)

    df['MA7'] = df['Price_Log'].rolling(7, min_periods=1).mean()
    df['MA14'] = df['Price_Log'].rolling(14, min_periods=1).mean()
    df['MA30'] = df['Price_Log'].rolling(30, min_periods=1).mean()

    df['Volatility7'] = df['Price_Log'].rolling(7, min_periods=1).std().fillna(0)
    df['Volatility14'] = df['Price_Log'].rolling(14, min_periods=1).std().fillna(0)

    df['Momentum'] = df['Price_Log'] - df['Price_Log'].shift(5)
    df['Momentum'] = df['Momentum'].fillna(0)

    return df.dropna()

# ======================
# HISTORY
# ======================
df_feat = create_features(df)
history = df_feat['Price_Log'].tolist()

# ======================
# RANDOM FOREST FORECAST (STABLE)
# ======================
def rf_forecast(history, steps):
    hist = history.copy()
    result = []

    for _ in range(steps):
        tmp = pd.DataFrame({"Price_Log": hist})
        tmp = create_features(tmp)

        x = tmp[features].iloc[-1:].values
        pred = rf.predict(x)[0]

        # stability clamp (VERY IMPORTANT)
        pred = np.clip(pred, -0.05, 0.05)

        next_val = tmp['Lag1'].iloc[-1] + pred
        hist.append(next_val)

        result.append(max(np.exp(next_val), 1))

    return result

# ======================
# GRADIENT BOOSTING FORECAST (STABLE)
# ======================
def gb_forecast(history, steps):
    hist = history.copy()
    result = []

    for _ in range(steps):
        tmp = pd.DataFrame({"Price_Log": hist})
        tmp = create_features(tmp)

        x = tmp[features].iloc[-1:].values
        pred = gb.predict(x)[0]

        pred = np.clip(pred, -0.05, 0.05)

        next_val = tmp['Lag1'].iloc[-1] + pred
        hist.append(next_val)

        result.append(max(np.exp(next_val), 1))

    return result

# ======================
# HYBRID (IMPROVED WEIGHTING)
# ======================
def hybrid_forecast(history, steps):
    rf_res = rf_forecast(history, steps)
    gb_res = gb_forecast(history, steps)

    return [0.6*r + 0.4*g for r, g in zip(rf_res, gb_res)]

# ======================
# RUN FORECAST
# ======================
if st.sidebar.button("Run Forecast"):

    if model_choice == "Random Forest":
        forecast = rf_forecast(history, days)

    elif model_choice == "Gradient Boosting":
        forecast = gb_forecast(history, days)

    elif model_choice == "Hybrid":
        forecast = hybrid_forecast(history, days)

    else:
        rf_f = rf_forecast(history, days)
        gb_f = gb_forecast(history, days)

        forecast = [(r + g) / 2 for r, g in zip(rf_f, gb_f)]

    # ======================
    # PLOT
    # ======================
    st.subheader("📈 10-Year Gold Price Forecast (Stable Version)")

    fig, ax = plt.subplots()
    ax.plot(forecast)
    ax.set_title("Gold Price Forecast")
    ax.set_xlabel("Days")
    ax.set_ylabel("Price")

    st.pyplot(fig)

    # ======================
    # DATA TABLE
    # ======================
    st.write(pd.DataFrame({"Forecast Price": forecast}))

    st.success("Forecast completed successfully!")

# ======================
# DATA PREVIEW
# ======================
st.subheader("Latest Data")
st.write(df.tail())

# ======================
# DASHBOARD IMAGES
# ======================
st.title("📊 Gold Market Analysis Dashboard")

tab1, tab2, tab3, tab4 = st.tabs([
    "Price Trends",
    "Volume Analysis",
    "Distribution & Correlation",
    "Model Performance"
])

with tab1:
    st.image(load_img("plot_graph/Daily Gold Price Trend from 2014 to 2026.png"))
    st.image(load_img("plot_graph/Historical price trends of gold (2014-2026).png"))
    st.image(load_img("plot_graph/Comparison of Open and Price Trajectories (2014-2026).png"))

with tab2:
    st.image(load_img("plot_graph/Average Trading Volume Comparison.png"))
    st.image(load_img("plot_graph/Gold Trading Volume Over Date.png"))

with tab3:
    st.image(load_img("plot_graph/Distribution of Daily Gold Prices.png"))
    st.image(load_img("plot_graph/Correlation between Open and Price.png"))

with tab4:
    st.image(load_img("plot_graph/rmse_comparison.png"))
    st.image(load_img("plot_graph/R2_score_comparison.png"))
    st.image(load_img("plot_graph/model_performance_table.png"))
