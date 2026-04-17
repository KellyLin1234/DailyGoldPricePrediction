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

# ======================
# 🎨 CUSTOM GOLD THEME UI
# ======================
st.markdown("""
    <style>
        .main {
            background-color: #0E1117;
        }

        h1, h2, h3 {
            color: #D4AF37;
        }

        .stMetric {
            background-color: #1C1F26;
            padding: 15px;
            border-radius: 12px;
        }

        .stButton>button {
            background-color: #D4AF37;
            color: black;
            font-weight: bold;
            border-radius: 10px;
        }

        .stSidebar {
            background-color: #11151C;
        }
    </style>
""", unsafe_allow_html=True)

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
st.sidebar.title("📊 Forecast Settings")
st.sidebar.markdown("### Model Control Panel")

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

forecast = None

if st.sidebar.button("🚀 Run Forecast"):

    # ======================
    # GENERATE FORECAST
    # ======================
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
    # SAFETY CHECK (IMPORTANT)
    # ======================
    if forecast is not None:

        forecast = smooth(forecast, 20)

        # ======================
        # KPI METRICS
        # ======================
        st.subheader("📌 Market Summary")

        col1, col2, col3 = st.columns(3)
        col1.metric("Latest Price", f"${df['Price'].iloc[-1]:,.2f}")
        col2.metric("Max Price", f"${df['Price'].max():,.2f}")
        col3.metric("Min Price", f"${df['Price'].min():,.2f}")

        trend = "📈 Upward Trend" if forecast[-1] > forecast[0] else "📉 Downward Trend"

        st.markdown(f"""
        ### 📊 Market Insight
        - Trend: **{trend}**
        - Forecast Horizon: **{years} years**
        - Model: **{model_choice}**
        """)

        # ======================
        # DAILY MODE
        # ======================
        if mode == "Daily Forecast":

            st.subheader("📈 Smoothed Forecast")

            fig, ax = plt.subplots(figsize=(10,5))
            ax.plot(forecast, linewidth=2, color="#D4AF37")
            ax.set_title("Gold Price Forecast")
            ax.set_xlabel("Days")
            ax.set_ylabel("Price")

            st.pyplot(fig)

            st.dataframe(pd.DataFrame({"Forecast": forecast}))

        # ======================
        # YEARLY MODE
        # ======================
        else:

            yearly_df = convert_to_yearly(forecast)

            st.subheader("📊 Yearly Financial Forecast")

            fig, ax = plt.subplots()
            ax.plot(yearly_df["Year"], yearly_df["Average Gold Price"],
                    marker="o", color="#D4AF37")

            ax.set_title("Yearly Gold Price Trend")
            ax.set_xlabel("Year")
            ax.set_ylabel("Price")

            st.pyplot(fig)

            st.dataframe(yearly_df)

        st.success("Forecast completed successfully!"))

    # ======================
    # KPI METRICS
    # ======================
    st.subheader("📌 Market Summary")

    col1, col2, col3 = st.columns(3)

    col1.metric("Latest Price", f"${df['Price'].iloc[-1]:,.2f}")
    col2.metric("Max Price", f"${df['Price'].max():,.2f}")
    col3.metric("Min Price", f"${df['Price'].min():,.2f}")

    # trend insight
    trend = "📈 Upward Trend" if forecast[-1] > forecast[0] else "📉 Downward Trend"

    st.markdown(f"""
    ### 📊 Market Insight
    - Trend: **{trend}**
    - Forecast Horizon: **{years} years**
    - Model: **{model_choice}**
    """)

    # ======================
    # DAILY MODE
    # ======================
    if mode == "Daily Forecast":

        st.subheader("📈 Smoothed 10-Year Daily Forecast")

        fig, ax = plt.subplots(figsize=(10,5))
        ax.plot(forecast, linewidth=2, color="#D4AF37")
        ax.set_title("Gold Price Forecast")
        ax.set_xlabel("Days")
        ax.set_ylabel("Price")

        st.pyplot(fig)

        st.dataframe(pd.DataFrame({"Forecast": forecast}))

    # ======================
    # YEARLY MODE
    # ======================
    else:

        yearly_df = convert_to_yearly(forecast)

        st.subheader("📊 Yearly Financial Forecast")

        fig, ax = plt.subplots()
        ax.plot(yearly_df["Year"], yearly_df["Average Gold Price"], marker="o", color="#D4AF37")
        ax.set_title("Yearly Gold Price Trend")
        ax.set_xlabel("Year")
        ax.set_ylabel("Price")

        st.pyplot(fig)

        st.dataframe(yearly_df)

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
