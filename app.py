import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from PIL import Image

# ======================
# PAGE CONFIG
# ======================
st.set_page_config(page_title="Gold Forecast Dashboard", layout="wide")

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
# ======================
# SIDEBAR
# ======================
years = st.sidebar.slider("Forecast Years", 1, 10, 10)
days = years * 365

model_choice = st.sidebar.selectbox(
    "Model",
    ["Random Forest", "Gradient Boosting", "Hybrid"]
)

mode = st.sidebar.radio(
    "Forecast Mode",
    ["Daily Forecast", "Yearly Forecast"]
)

# ======================
# FEATURE ENGINEERING
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

df_feat = create_features(df)
history = df_feat['Price_Log'].tolist()

# ======================
# FORECAST FUNCTIONS (IMPORTANT FIX)
# ======================

def rf_forecast(history, steps):
    hist = history.copy()
    result = []

    for _ in range(steps):
        tmp = pd.DataFrame({"Price_Log": hist})
        tmp = create_features(tmp)

        x = tmp[features].iloc[-1:].values
        pred = rf.predict(x)[0]

        # 🔥 stabilize prediction
        pred = np.clip(pred, -0.02, 0.02)

        next_val = tmp['Lag1'].iloc[-1] + pred

        # safety clamp (prevents explosion)
        next_val = np.clip(next_val, np.log(500), np.log(5000))

        hist.append(next_val)
        result.append(np.exp(next_val))

    return result

def gb_forecast(history, steps):
    hist = history.copy()
    result = []

    for _ in range(steps):
        tmp = pd.DataFrame({"Price_Log": hist})
        tmp = create_features(tmp)

        x = tmp[features].iloc[-1:].values
        pred = gb.predict(x)[0]

        pred = np.clip(pred, -0.02, 0.02)

        next_val = tmp['Lag1'].iloc[-1] + pred
        next_val = np.clip(next_val, np.log(500), np.log(5000))

        hist.append(next_val)
        result.append(np.exp(next_val))

    return result


def hybrid_forecast(history, steps):
    rf_res = rf_forecast(history, steps)
    gb_res = gb_forecast(history, steps)

    # weighted average (more stable than 0.5/0.5)
    return [0.6*r + 0.4*g for r, g in zip(rf_res, gb_res)]

# ======================
# YEARLY CONVERSION
# ======================
def convert_to_yearly(forecast, start_year=2026):
    yearly = []

    for i in range(0, len(forecast), 365):
        chunk = forecast[i:i+365]
        yearly.append(np.mean(chunk))

    years_list = list(range(start_year, start_year + len(yearly)))

    return pd.DataFrame({
        "Year": years_list,
        "Average Gold Price": yearly
    })

# ======================
# RUN FORECAST
# ======================
if st.sidebar.button("🚀 Run Forecast"):

    # FIX: ALWAYS DEFINE FORECAST FIRST
    forecast = None

    if model_choice == "Random Forest":
        forecast = rf_forecast(history, days)

    elif model_choice == "Gradient Boosting":
        forecast = gb_forecast(history, days)

    elif model_choice == "Hybrid":
        forecast = hybrid_forecast(history, days)

    # ======================
    # SAFETY CHECK
    # ======================
    if forecast is not None:

        # smoothing
        forecast = pd.Series(forecast).rolling(20, min_periods=1).mean().tolist()

        # ======================
        # KPI
        # ======================
        st.subheader("📊 Market Overview")

        col1, col2, col3 = st.columns(3)
        col1.metric("Latest Price", f"${df['Price'].iloc[-1]:,.2f}")
        col2.metric("Max Price", f"${df['Price'].max():,.2f}")
        col3.metric("Min Price", f"${df['Price'].min():,.2f}")

        trend = "📈 Upward" if forecast[-1] > forecast[0] else "📉 Downward"

        st.markdown(f"""
        ### Market Insight
        - Trend: **{trend}**
        - Model: **{model_choice}**
        - Horizon: **{years} years**
        """)

        # ======================
        # DAILY
        # ======================
        if mode == "Daily Forecast":

            fig, ax = plt.subplots(figsize=(10,5))
            ax.plot(forecast, color="gold")
            ax.set_title("10-Year Gold Price Forecast")
            st.pyplot(fig)

            st.dataframe(pd.DataFrame({"Forecast": forecast}))

        # ======================
        # YEARLY
        # ======================
        else:

            yearly = convert_to_yearly(forecast)

            fig, ax = plt.subplots()
            ax.plot(yearly["Year"], yearly["Average Gold Price"], marker="o")
            ax.set_title("Yearly Forecast")
            st.pyplot(fig)

            st.dataframe(yearly)

        st.success("Forecast completed successfully!")

# ======================
# DATA PREVIEW
# ======================
st.subheader("📁 Latest Data")
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

    st.image("plot_graph/Daily Gold Price Trend from 2014 to 2026.png")
    st.image("plot_graph/Historical price trends of gold (2014-2026).png")
    st.image("plot_graph/Comparison of Open and Price Trajectories (2014-2026).png")
    st.image("plot_graph/Daily Opening Price Fluctuations (2014-2026).png")
    st.image("plot_graph/Daily Highest Price of Gold (2014–2026).png")
    st.image("plot_graph/Daily Lowest Price of Gold (2014–2026).png")

# ======================
# TAB 2 - VOLUME ANALYSIS
# ======================
with tab2:
    st.subheader("Trading Volume Analysis")

    st.image("plot_graph/Average Trading Volume Comparison.png")
    st.image("plot_graph/Gold Trading Volume Over Date.png")
    st.image("plot_graph/Gold Trading Volume Over Year.png")
    st.image("plot_graph/Relationship Between Volume and Gold Price.png")

# ======================
# TAB 3 - DISTRIBUTION & CORRELATION
# ======================
with tab3:
    st.subheader("Statistical Analysis")

    st.image("plot_graph/Distribution of Daily Gold Prices.png")
    st.image("plot_graph/Distribution of Gold Price Percentage Change.png")
    st.image("plot_graph/Correlation between Open and Price.png")
    st.image("plot_graph/Market Volatility Comparison.png")
    st.image("plot_graph/Gold Price Time Series with Holiday Events.png")

# ======================
# TAB 4 - MODEL PERFORMANCE
# ======================
with tab4:
    st.subheader("Model Evaluation Results")

    st.image("plot_graph/mae_comparison.png")
    st.image("plot_graph/mape_comparison.png")
    st.image("plot_graph/rmse_comparison.png")
    st.image("plot_graph/R2_score_comparison.png")
    st.image("plot_graph/model_performance_table.png")
