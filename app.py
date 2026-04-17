import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import os

# ======================================================
# CONFIG
# ======================================================
st.set_page_config(page_title="Gold Intelligence Dashboard", layout="wide")
st.title("💰 Gold Price Prediction Dashboard")

# ======================================================
# DATA LOADING
# ======================================================
@st.cache_data
def load_data():
    df = pd.read_csv("Gold Price.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values("Date").reset_index(drop=True)

    if 'LSTM_Pred' not in df.columns:
        df['LSTM_Pred'] = 0

    return df

df = load_data()

# ======================================================
# MODEL LOADING
# ======================================================
@st.cache_resource
def load_models():
    models = {}

    models["hybrid"] = joblib.load("models/hybrid_rf.pkl")
    models["rf"] = joblib.load("models/random_forest.pkl")
    models["gb"] = joblib.load("models/gradient_boosting.pkl")

    return models

models = load_models()

# ======================================================
# SIDEBAR
# ======================================================
st.sidebar.header("Forecast Settings")

years = st.sidebar.slider("Forecast Horizon (Years)", 1, 10, 5)
n_days = years * 365

# ======================================================
# KPIS
# ======================================================
def show_kpis(df):
    last_price = df['Price'].iloc[-1]

    col1, col2, col3 = st.columns(3)
    col1.metric("Current Gold Price", f"{last_price:,.2f}")
    col2.metric("Models", "RF / GB / Hybrid")
    col3.metric("Forecast Horizon", f"{years} Years")

show_kpis(df)

# ======================================================
# HISTORICAL PLOT
# ======================================================
def plot_history(df):
    st.subheader("📈 Historical Gold Price")

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(df['Date'], df['Price'])
    ax.set_title("Gold Price History")

    st.pyplot(fig)

plot_history(df)

# ======================================================
# PERFORMANCE IMAGES
# ======================================================
def show_model_performance():
    st.subheader("📊 Model Performance Comparison")

    if not os.path.exists("plot_graph"):
        st.warning("Performance graphs not found.")
        return

    tab1, tab2, tab3 = st.tabs(["Table", "Error Metrics", "R² Scores"])

    with tab1:
        st.image("plot_graph/model_performance_table.png", use_container_width=True)

    with tab2:
        st.image("plot_graph/rmse_comparison.png")
        st.image("plot_graph/mae_comparison.png")
        st.image("plot_graph/mape_comparison.png")

    with tab3:
        st.image("plot_graph/R2 Score Comparison_comparison.png")

show_model_performance()

# ======================================================
# FEATURE ENGINEERING
# ======================================================
def create_features(prices, lstm_series):
    lag1 = prices[-1]
    lag2 = prices[-2]
    ma7 = np.mean(prices[-7:])
    lstm_val = np.mean(lstm_series[-30:])

    return lag1, lag2, ma7, lstm_val

# ======================================================
# FORECAST ENGINE
# ======================================================
def run_forecast(models, df, n_days):
    st.subheader("🔮 Forecast Comparison")

    prices = df['Price'].tolist()
    lstm_series = df['LSTM_Pred'].tolist()

    rf_preds = []
    gb_preds = []
    hybrid_preds = []

    lag1, lag2 = prices[-1], prices[-2]
    history = prices[-30:].copy()

    for _ in range(n_days):

        ma7 = np.mean(history[-7:])
        lstm_val = np.mean(lstm_series[-30:])

        # -------------------
        # Hybrid
        # -------------------
        X_hybrid = np.array([[lag1, lag2, ma7, lstm_val]])
        hybrid_pred = models["hybrid"].predict(X_hybrid)[0]
        hybrid_preds.append(hybrid_pred)

        # -------------------
        # Random Forest
        # -------------------
        rf_model = models["rf"]
        rf_features = list(rf_model.feature_names_in_)

        rf_input = pd.DataFrame([[lag1, lag2, ma7, lstm_val]],
                                columns=['lag1', 'lag2', 'ma7', 'LSTM_Pred'])

        rf_input = rf_input.reindex(columns=rf_features, fill_value=0)
        rf_pred = rf_model.predict(rf_input)[0]
        rf_preds.append(rf_pred)

        # -------------------
        # Gradient Boosting
        # -------------------
        gb_model = models["gb"]
        gb_features = list(gb_model.feature_names_in_)

        gb_input = rf_input.reindex(columns=gb_features, fill_value=0)
        gb_pred = gb_model.predict(gb_input)[0]
        gb_preds.append(gb_pred)

        # -------------------
        # UPDATE
        # -------------------
        lag2 = lag1
        lag1 = hybrid_pred
        history.append(hybrid_pred)

    return rf_preds, gb_preds, hybrid_preds

rf_preds, gb_preds, hybrid_preds = run_forecast(models, df, n_days)

# ======================================================
# FUTURE DATES
# ======================================================
future_dates = pd.date_range(
    start=df['Date'].iloc[-1] + pd.Timedelta(days=1),
    periods=n_days
)

# ======================================================
# FORECAST VISUALIZATION
# ======================================================
def plot_forecast(dates, rf, gb, hybrid):
    fig, ax = plt.subplots(figsize=(14, 5))

    ax.plot(dates, hybrid, label="Hybrid (Best)", linewidth=3)
    ax.plot(dates, rf, label="Random Forest", alpha=0.7)
    ax.plot(dates, gb, label="Gradient Boosting", alpha=0.7)

    ax.legend()
    ax.set_title("Gold Price Forecast Comparison")

    st.pyplot(fig)

plot_forecast(future_dates, rf_preds, gb_preds, hybrid_preds)

# ======================================================
# SUMMARY TABLE
# ======================================================
def show_summary(rf, gb, hybrid):
    st.subheader("📊 Forecast Summary")

    summary = pd.DataFrame({
        "Model": ["Hybrid", "Random Forest", "Gradient Boosting"],
        "Final Price": [hybrid[-1], rf[-1], gb[-1]]
    })

    st.dataframe(summary, use_container_width=True)

show_summary(rf_preds, gb_preds, hybrid_preds)

# ======================================================
# GROWTH ANALYSIS
# ======================================================
def show_growth(hybrid, years):
    start = hybrid[0]
    end = hybrid[-1]

    pct = ((end - start) / start) * 100

    if pct > 0:
        st.success(f"📈 Expected Growth: +{pct:.2f}% over {years} years")
    else:
        st.warning(f"📉 Expected Decline: {pct:.2f}% over {years} years")

show_growth(hybrid_preds, years)
