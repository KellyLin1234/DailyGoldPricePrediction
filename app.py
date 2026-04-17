import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# ======================
# PAGE CONFIG
# ======================
st.set_page_config(page_title="Gold Forecast Dashboard", layout="wide")

# ======================
# CACHE DATA (VERY IMPORTANT SPEED FIX)
# ======================
@st.cache_data
def load_data():
    df = pd.read_csv("Gold Price.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')

    df['Price'] = df['Price'].ffill()
    df['Price_Log'] = np.log(df['Price'])

    return df

df = load_data()

# ======================
# LOAD MODELS (CACHE OPTIONAL)
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
    "Model",
    ["Random Forest", "Gradient Boosting", "Hybrid"]
)

mode = st.sidebar.radio(
    "Forecast Mode",
    ["Daily Forecast", "Yearly Forecast"]
)

# ======================
# PRECOMPUTE FEATURES ONCE (IMPORTANT SPEED FIX)
# ======================
def create_features(df):
    df = df.copy()

    df['Lag1'] = df['Price_Log'].shift(1)
    df['Lag2'] = df['Price_Log'].shift(2)
    df['Lag3'] = df['Price_Log'].shift(3)
    df['Lag5'] = df['Price_Log'].shift(5)
    df['Lag10'] = df['Price_Log'].shift(10)

    df['MA7'] = df['Price_Log'].rolling(7).mean()
    df['MA14'] = df['Price_Log'].rolling(14).mean()
    df['MA30'] = df['Price_Log'].rolling(30).mean()

    df['Volatility7'] = df['Price_Log'].rolling(7).std()
    df['Volatility14'] = df['Price_Log'].rolling(14).std()

    df['Momentum'] = df['Price_Log'] - df['Price_Log'].shift(5)

    return df.dropna()

df_feat = create_features(df)
history = df_feat['Price_Log'].tolist()

# ======================
# FAST FORECAST CORE (NO REBUILD INSIDE LOOP)
# ======================
def forecast_model(model, history, steps):
    hist = history.copy()
    result = []

    for _ in range(steps):
        tmp = pd.DataFrame({"Price_Log": hist})

        tmp['Lag1'] = tmp['Price_Log'].shift(1)
        tmp['Lag2'] = tmp['Price_Log'].shift(2)
        tmp['Lag3'] = tmp['Price_Log'].shift(3)
        tmp['Lag5'] = tmp['Price_Log'].shift(5)
        tmp['Lag10'] = tmp['Price_Log'].shift(10)

        tmp['MA7'] = tmp['Price_Log'].rolling(7).mean()
        tmp['MA14'] = tmp['Price_Log'].rolling(14).mean()
        tmp['MA30'] = tmp['Price_Log'].rolling(30).mean()

        tmp['Volatility7'] = tmp['Price_Log'].rolling(7).std()
        tmp['Volatility14'] = tmp['Price_Log'].rolling(14).std()

        tmp['Momentum'] = tmp['Price_Log'] - tmp['Price_Log'].shift(5)

        tmp = tmp.dropna()

        x = tmp[features].iloc[-1:].values
        pred = model.predict(x)[0]

        pred = np.clip(pred, -0.02, 0.02)

        next_val = tmp['Lag1'].iloc[-1] + pred
        hist.append(next_val)

        result.append(np.exp(next_val))

    return result

# ======================
# WRAPPERS
# ======================
def rf_forecast(history, steps):
    return forecast_model(rf, history, steps)

def gb_forecast(history, steps):
    return forecast_model(gb, history, steps)

def hybrid_forecast(history, steps):
    rf_res = rf_forecast(history, steps)
    gb_res = gb_forecast(history, steps)
    return [(r + g) / 2 for r, g in zip(rf_res, gb_res)]

# ======================
# YEARLY CONVERSION (FAST)
# ======================
def convert_to_yearly(forecast, start_year=2026):
    forecast = np.array(forecast)

    yearly = forecast.reshape(-1, 365).mean(axis=1)

    years = np.arange(start_year, start_year + len(yearly))

    return pd.DataFrame({
        "Year": years,
        "Avg Price": yearly
    })

# ======================
# RUN FORECAST (OPTIMIZED)
# ======================
if st.sidebar.button("🚀 Run Forecast"):

    with st.spinner("Generating forecast..."):

        if model_choice == "Random Forest":
            forecast = rf_forecast(history, days)

        elif model_choice == "Gradient Boosting":
            forecast = gb_forecast(history, days)

        else:
            forecast = hybrid_forecast(history, days)

        # smoothing (fast version)
        forecast = pd.Series(forecast).rolling(15, min_periods=1).mean().tolist()

        # ======================
        # KPI
        # ======================
        st.subheader("📊 Market Overview")

        col1, col2, col3 = st.columns(3)
        col1.metric("Latest", f"${df['Price'].iloc[-1]:,.2f}")
        col2.metric("Max", f"${df['Price'].max():,.2f}")
        col3.metric("Min", f"${df['Price'].min():,.2f}")

        st.success("Forecast generated successfully")

        # ======================
        # PLOT
        # ======================
        fig, ax = plt.subplots()
        ax.plot(forecast, color="gold")
        ax.set_title("Gold Forecast")
        st.pyplot(fig)

        # ======================
        # YEARLY
        # ======================
        if mode == "Yearly Forecast":
            yearly = convert_to_yearly(forecast)
            st.dataframe(yearly)

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
