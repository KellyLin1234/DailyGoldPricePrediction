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
# FAST YEARLY FORECAST ENGINE
# ======================
def yearly_forecast(model, history, years):
    hist = history.copy()
    predictions = []

    steps = years  # yearly forecast = 1 step per year

    for _ in range(steps):

        tmp = pd.DataFrame({"Price_Log": hist})
        tmp = create_features(tmp)

        x = tmp[features].iloc[-1:].values

        pred = model.predict(x)[0]
        pred = np.clip(pred, -0.03, 0.03)

        next_val = tmp['Lag1'].iloc[-1] + pred
        hist.append(next_val)

        predictions.append(np.exp(next_val))

    return predictions


def hybrid_forecast(history, years):
    rf_pred = yearly_forecast(rf, history, years)
    gb_pred = yearly_forecast(gb, history, years)

    return [(r + g) / 2 for r, g in zip(rf_pred, gb_pred)]


# ======================
# SIDEBAR
# ======================
years = st.sidebar.slider("Forecast Years", 1, 10, 10)

model_choice = st.sidebar.selectbox(
    "Model",
    ["Random Forest", "Gradient Boosting", "Hybrid"]
)

# ======================
# RUN FORECAST
# ======================
if st.sidebar.button("🚀 Run Forecast"):

    if model_choice == "Random Forest":
        forecast = yearly_forecast(rf, history, years)

    elif model_choice == "Gradient Boosting":
        forecast = yearly_forecast(gb, history, years)

    else:
        forecast = hybrid_forecast(history, years)

    # ======================
    # DISPLAY
    # ======================
    st.subheader("📊 Yearly Gold Price Forecast (FAST MODE)")

    year_labels = list(range(2026, 2026 + years))

    fig, ax = plt.subplots()
    ax.plot(year_labels, forecast, marker="o")
    ax.set_title("Gold Price Forecast (Yearly)")
    ax.set_xlabel("Year")
    ax.set_ylabel("Price")

    st.pyplot(fig)

    st.dataframe(pd.DataFrame({
        "Year": year_labels,
        "Forecast Price": forecast
    }))

# ======================
# DATA PREVIEW
# ======================
st.subheader("📁 Latest Data")
st.dataframe(df.tail())

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
    st.image("plot_graph/Daily Gold Price Trend from 2014 to 2026.png")
    st.image("plot_graph/Historical price trends of gold (2014-2026).png")
    st.image("plot_graph/Comparison of Open and Price Trajectories (2014-2026).png")
    st.image("plot_graph/Daily Opening Price Fluctuations (2014-2026).png")
    st.image("plot_graph/Daily Highest Price of Gold (2014–2026).png")
    st.image("plot_graph/Daily Lowest Price of Gold (2014–2026).png")

with tab2:
    st.image("plot_graph/Average Trading Volume Comparison.png")
    st.image("plot_graph/Gold Trading Volume Over Date.png")
    st.image("plot_graph/Gold Trading Volume Over Year.png")
    st.image("plot_graph/Relationship Between Volume and Gold Price.png")

with tab3:
    st.image("plot_graph/Distribution of Daily Gold Prices.png")
    st.image("plot_graph/Distribution of Gold Price Percentage Change.png")
    st.image("plot_graph/Correlation between Open and Price.png")
    st.image("plot_graph/Market Volatility Comparison.png")
    st.image("plot_graph/Gold Price Time Series with Holiday Events.png")

with tab4:
    st.image("plot_graph/mae_comparison.png")
    st.image("plot_graph/mape_comparison.png")
    st.image("plot_graph/rmse_comparison.png")
    st.image("plot_graph/R2_score_comparison.png")
    st.image("plot_graph/model_performance_table.png")
