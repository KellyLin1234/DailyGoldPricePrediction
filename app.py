import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# ======================
# PAGE CONFIG
# ======================
st.set_page_config(page_title="Gold Forecast (Yearly Fast Mode)", layout="wide")

# ======================
# LOAD DATA (FAST)
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

model_choice = st.sidebar.selectbox(
    "Model",
    ["Random Forest", "Gradient Boosting", "Hybrid"]
)

# ======================
# FEATURE ENGINEERING (ONLY FOR LAST STATE)
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

# ======================
# GET LAST STATE ONLY (KEY SPEED FIX)
# ======================
history = df_feat['Price_Log'].tolist()

# ======================
# FAST YEARLY FORECAST (NO DAILY LOOP)
# ======================
def yearly_forecast(model, history, years):
    hist = history.copy()
    result = []

    for _ in range(years):

        latest = build_latest_features(hist)

        x = latest[features].values.reshape(1, -1)

        # safety check (IMPORTANT)
        if x.shape[1] != len(features):
            raise ValueError(f"Feature mismatch: {x.shape[1]} vs {len(features)}")

        pred = model.predict(x)[0]
        pred = np.clip(pred, -0.03, 0.03)

        next_val = latest['Lag1'] + pred
        hist.append(next_val)

        result.append(np.exp(next_val))

    return result

# ======================
# MODEL WRAPPERS
# ======================
def rf_forecast(history, years):
    return yearly_forecast(rf, history, years)

def gb_forecast(history, years):
    return yearly_forecast(gb, history, years)

def hybrid_forecast(history, years):
    rf_res = rf_forecast(history, years)
    gb_res = gb_forecast(history, years)
    return [(r + g) / 2 for r, g in zip(rf_res, gb_res)]

# ======================
# RUN FORECAST
# ======================
if st.sidebar.button("🚀 Generate Forecast"):

    with st.spinner("Generating yearly forecast..."):

        if model_choice == "Random Forest":
            forecast = rf_forecast(history, years)

        elif model_choice == "Gradient Boosting":
            forecast = gb_forecast(history, years)

        else:
            forecast = hybrid_forecast(history, years)

        # ======================
        # YEAR LABELS
        # ======================
        start_year = df['Date'].dt.year.max() + 1
        years_index = list(range(start_year, start_year + len(forecast)))

        yearly_df = pd.DataFrame({
            "Year": years_index,
            "Predicted Gold Price": forecast
        })

        # ======================
        # METRICS
        # ======================
        st.subheader("📊 Forecast Summary")

        col1, col2, col3 = st.columns(3)
        col1.metric("Latest Price", f"${df['Price'].iloc[-1]:,.2f}")
        col2.metric("Forecast Start", f"{start_year}")
        col3.metric("Years Predicted", f"{years}")

        # ======================
        # PLOT
        # ======================
        st.subheader("📈 Yearly Gold Price Forecast")

        fig, ax = plt.subplots()
        ax.plot(yearly_df["Year"], yearly_df["Predicted Gold Price"],
                marker="o", color="gold", linewidth=3)

        ax.set_title("Gold Price Forecast (Yearly)")
        ax.set_xlabel("Year")
        ax.set_ylabel("Price")

        st.pyplot(fig)

        # ======================
        # TABLE
        # ======================
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
