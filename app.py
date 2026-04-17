import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from PIL import Image
from collections import deque

# ======================
# PAGE CONFIG
# ======================
st.set_page_config(page_title="Gold Forecast Dashboard", layout="wide")

# ======================
# CACHE DATA & MODELS
# ======================
@st.cache_data
def load_data():
    df = pd.read_csv("Gold Price.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    df['Price'] = df['Price'].ffill()
    df['Price_Log'] = np.log(df['Price'])
    return df

@st.cache_resource
def load_models():
    rf = joblib.load("models/random_forest_log.pkl")
    gb = joblib.load("models/gradient_boosting_log.pkl")
    features = joblib.load("models/features.pkl")
    return rf, gb, features

df = load_data()
rf, gb, features = load_models()

history_prices = df['Price_Log'].tolist()

# ======================
# THEME
# ======================
st.markdown("""
<style>
.main { background-color: #0E1117; }
h1,h2,h3 { color: #D4AF37; }
.stButton>button { background:#D4AF37;color:black;font-weight:bold; }
</style>
""", unsafe_allow_html=True)

# ======================
# SIDEBAR
# ======================
st.sidebar.title("📊 Forecast Settings")

years = st.sidebar.slider("Forecast Years", 1, 10, 5)
steps = years * 365

model_choice = st.sidebar.selectbox(
    "Model",
    ["Random Forest", "Gradient Boosting", "Hybrid", "All"]
)

mode = st.sidebar.radio(
    "Forecast Mode",
    ["Daily Forecast", "Yearly Forecast"]
)

# ======================
# FAST FEATURE ENGINEERING (NO DATAFRAME)
# ======================
def get_features(hist, i):
    """
    Compute features only from list (FAST)
    """
    def get(idx):
        return hist[idx] if idx >= 0 else hist[0]

    price = hist[i]

    lag1 = get(i-1)
    lag2 = get(i-2)
    lag3 = get(i-3)
    lag5 = get(i-5)
    lag10 = get(i-10)

    window7 = hist[max(0, i-7):i+1]
    window14 = hist[max(0, i-14):i+1]

    ma7 = np.mean(window7)
    ma14 = np.mean(window14)

    vol7 = np.std(window7)
    vol14 = np.std(window14)

    momentum = price - get(i-5)

    return np.array([lag1, lag2, lag3, lag5, lag10,
                     ma7, ma14, vol7, vol14, momentum]).reshape(1, -1)

# ======================
# FAST FORECAST ENGINE
# ======================
def forecast(model, history, steps):
    hist = history.copy()
    result = []

    for i in range(len(hist), len(hist) + steps):

        x = get_features(hist, len(hist)-1)

        pred = model.predict(x)[0]
        pred = np.clip(pred, -0.03, 0.03)

        next_val = hist[-1] + pred
        hist.append(next_val)

        result.append(max(np.exp(next_val), 1))

    return result

# ======================
# HYBRID (FAST VERSION)
# ======================
def hybrid_forecast(history, steps):
    rf_res = forecast(rf, history, steps)
    gb_res = forecast(gb, history, steps)

    return [(0.6*r + 0.4*g) for r, g in zip(rf_res, gb_res)]

# ======================
# SMOOTH
# ======================
def smooth(series, window=20):
    return pd.Series(series).rolling(window, min_periods=1).mean().tolist()

# ======================
# YEARLY CONVERSION
# ======================
def to_yearly(forecast, start=2026):
    arr = np.array(forecast)

    yearly = []
    for i in range(0, len(arr), 365):
        yearly.append(np.median(arr[i:i+365]))

    return pd.DataFrame({
        "Year": list(range(start, start + len(yearly))),
        "Price": yearly
    })

# ======================
# RUN BUTTON
# ======================
if st.sidebar.button("🚀 Run Forecast"):

    if model_choice == "Random Forest":
        forecasted = forecast(rf, history_prices, steps)

    elif model_choice == "Gradient Boosting":
        forecasted = forecast(gb, history_prices, steps)

    elif model_choice == "Hybrid":
        forecasted = hybrid_forecast(history_prices, steps)

    else:
        rf_f = forecast(rf, history_prices, steps)
        gb_f = forecast(gb, history_prices, steps)
        forecasted = [(r+g)/2 for r, g in zip(rf_f, gb_f)]

    forecasted = smooth(forecasted, 20)

    # ======================
    # METRICS
    # ======================
    st.subheader("📌 Market Summary")

    col1, col2, col3 = st.columns(3)

    col1.metric("Latest", f"${df['Price'].iloc[-1]:,.2f}")
    col2.metric("Max", f"${df['Price'].max():,.2f}")
    col3.metric("Min", f"${df['Price'].min():,.2f}")

    trend = "📈 Up" if forecasted[-1] > forecasted[0] else "📉 Down"

    st.markdown(f"""
    ### Insight
    - Trend: **{trend}**
    - Horizon: **{years} years**
    - Model: **{model_choice}**
    """)

    # ======================
    # DAILY
    # ======================
    if mode == "Daily Forecast":

        st.subheader("📈 Forecast")

        fig, ax = plt.subplots()
        ax.plot(forecasted, color="#D4AF37")
        ax.set_title("Gold Forecast")
        st.pyplot(fig)

        st.dataframe(pd.DataFrame({"Forecast": forecasted}))

    # ======================
    # YEARLY
    # ======================
    else:

        yearly = to_yearly(forecasted)

        st.subheader("📊 Yearly Forecast")

        fig, ax = plt.subplots()
        ax.plot(yearly["Year"], yearly["Price"], marker="o", color="#D4AF37")
        st.pyplot(fig)

        st.dataframe(yearly)

    st.success("Done!")

# ======================
# DATA
# ======================
st.subheader("📁 Data Preview")
st.dataframe(df.tail())

# ======================
# EDA & VISUALIZATION DASHBOARD
# ======================
st.title("📊 Gold Market Analysis Dashboard")

tab1, tab2, tab3, tab4 = st.tabs([
    "Price Trends",
    "Volume Analysis",
    "Distribution & Correlation",
    "Model Performance"
])

# ======================
# TAB 1 - PRICE TRENDS
# ======================
with tab1:
    st.subheader("Gold Price Trends (2014–2026)")

    st.image(load_img("plot_graph/Daily Gold Price Trend from 2014 to 2026.png"))
    st.image(load_img("plot_graph/Historical price trends of gold (2014-2026).png"))
    st.image(load_img("plot_graph/Comparison of Open and Price Trajectories (2014-2026).png"))
    st.image(load_img("plot_graph/Daily Opening Price Fluctuations (2014-2026).png"))
    st.image(load_img("plot_graph/Daily Highest Price of Gold (2014–2026).png"))
    st.image(load_img("plot_graph/Daily Lowest Price of Gold (2014–2026).png"))

# ======================
# TAB 2 - VOLUME ANALYSIS
# ======================
with tab2:
    st.subheader("Trading Volume Analysis")

    st.image(load_img("plot_graph/Average Trading Volume Comparison.png"))
    st.image(load_img("plot_graph/Gold Trading Volume Over Date.png"))
    st.image(load_img("plot_graph/Gold Trading Volume Over Year.png"))
    st.image(load_img("plot_graph/Relationship Between Volume and Gold Price.png"))

# ======================
# TAB 3 - DISTRIBUTION & CORRELATION
# ======================
with tab3:
    st.subheader("Statistical Analysis")

    st.image(load_img("plot_graph/Distribution of Daily Gold Prices.png"))
    st.image(load_img("plot_graph/Distribution of Gold Price Percentage Change.png"))
    st.image(load_img("plot_graph/Correlation between Open and Price.png"))
    st.image(load_img("plot_graph/Market Volatility Comparison.png"))
    st.image(load_img("plot_graph/Gold Price Time Series with Holiday Events.png"))

# ======================
# TAB 4 - MODEL PERFORMANCE
# ======================
with tab4:
    st.subheader("Model Evaluation Results")

    st.image(load_img("plot_graph/mae_comparison.png"))
    st.image(load_img("plot_graph/mape_comparison.png"))
    st.image(load_img("plot_graph/rmse_comparison.png"))
    st.image(load_img("plot_graph/R2_score_comparison.png"))
    st.image(load_img("plot_graph/model_performance_table.png"))
