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
    "Model",
    ["Random Forest", "Gradient Boosting", "Hybrid", "All"]
)

mode = st.sidebar.radio(
    "Forecast Mode",
    ["Daily Forecast", "Yearly Financial Forecast"]
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
# SMOOTH FUNCTION
# ======================
def smooth(series, window=20):
    return pd.Series(series).rolling(window, min_periods=1).mean().tolist()

# ======================
# RF FORECAST
# ======================
def rf_forecast(history, steps):
    hist = history.copy()
    result = []

    for _ in range(steps):
        tmp = pd.DataFrame({"Price_Log": hist})
        tmp = create_features(tmp)

        x = tmp[features].iloc[-1:].values
        pred = rf.predict(x)[0]

        pred = np.clip(pred, -0.03, 0.03)

        next_val = tmp['Lag1'].iloc[-1] + pred
        hist.append(next_val)

        result.append(max(np.exp(next_val), 1))

    return result

# ======================
# GB FORECAST
# ======================
def gb_forecast(history, steps):
    hist = history.copy()
    result = []

    for _ in range(steps):
        tmp = pd.DataFrame({"Price_Log": hist})
        tmp = create_features(tmp)

        x = tmp[features].iloc[-1:].values
        pred = gb.predict(x)[0]

        pred = np.clip(pred, -0.03, 0.03)

        next_val = tmp['Lag1'].iloc[-1] + pred
        hist.append(next_val)

        result.append(max(np.exp(next_val), 1))

    return result

# ======================
# HYBRID
# ======================
def hybrid_forecast(history, steps):
    rf_res = rf_forecast(history, steps)
    gb_res = gb_forecast(history, steps)

    return [0.6*r + 0.4*g for r, g in zip(rf_res, gb_res)]

# ======================
# YEARLY CONVERSION
# ======================
def convert_to_yearly(forecast, start_year=2026):
    forecast = np.array(forecast)

    yearly = []

    for i in range(0, len(forecast), 365):
        chunk = forecast[i:i+365]

        if len(chunk) > 0:
            yearly.append(np.median(chunk))

    years = list(range(start_year, start_year + len(yearly)))

    return pd.DataFrame({
        "Year": years,
        "Average Gold Price": yearly
    })

# ======================
# RUN FORECAST
# ======================
if st.sidebar.button("Run Forecast"):

    # generate forecast
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

    # SMOOTH FORECAST (IMPORTANT FIX)
    forecast = smooth(forecast, window=20)

    # ======================
    # DAILY MODE
    # ======================
    if mode == "Daily Forecast":

        st.subheader("📈 Smoothed 10-Year Daily Gold Forecast")

        fig, ax = plt.subplots()
        ax.plot(forecast, linewidth=2)
        ax.set_title("Gold Price Forecast")
        ax.set_xlabel("Days")
        ax.set_ylabel("Price")

        st.pyplot(fig)

        st.write(pd.DataFrame({"Forecast": forecast}))

    # ======================
    # YEARLY MODE
    # ======================
    else:

        yearly_df = convert_to_yearly(forecast)

        st.subheader("📊 Yearly Financial Forecast")

        fig, ax = plt.subplots()
        ax.plot(yearly_df["Year"], yearly_df["Average Gold Price"], marker="o")
        ax.set_title("Yearly Gold Price Trend")
        ax.set_xlabel("Year")
        ax.set_ylabel("Price")

        st.pyplot(fig)

        st.write(yearly_df)

    st.success("Forecast completed successfully!")

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
