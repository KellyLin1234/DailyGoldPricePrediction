import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import os

# ======================
# PAGE CONFIG
# ======================
st.set_page_config(page_title="Gold Intelligence Dashboard", layout="wide")
st.title("💰 Gold Price Prediction Dashboard")

# ======================
# LOAD DATA
# ======================
@st.cache_data
def load_data():
    df = pd.read_csv("Gold Price.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values("Date").reset_index(drop=True)

    if 'LSTM_Pred' not in df.columns:
        df['LSTM_Pred'] = 0

    return df

df = load_data()

# ======================
# LOAD MODELS (SAFE)
# ======================
def load_model(path):
    try:
        return joblib.load(path)
    except:
        return None

hybrid_model = load_model("models/hybrid_rf.pkl")
rf_model = load_model("models/random_forest.pkl")
gb_model = load_model("models/gradient_boosting.pkl")

# ======================
# SIDEBAR
# ======================
st.sidebar.header("Forecast Settings")
years = st.sidebar.slider("Forecast Horizon (Years)", 1, 10, 5)
n_days = years * 365

# ======================
# KPI
# ======================
last_price = df['Price'].iloc[-1]

col1, col2, col3 = st.columns(3)
col1.metric("Current Gold Price", f"{last_price:,.2f}")
col2.metric("Models", "RF / GB / Hybrid")
col3.metric("Forecast Horizon", f"{years} Years")

# ======================
# HISTORICAL
# ======================
st.subheader("📈 Historical Gold Price")

fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(df['Date'], df['Price'], linewidth=2)
ax.set_title("Gold Price History")
st.pyplot(fig)

# ======================
# PERFORMANCE IMAGES
# ======================
st.subheader("📊 Model Performance")

if os.path.exists("plot_graph"):
    tab1, tab2, tab3 = st.tabs(["📋 Table", "📉 Errors", "📈 Scores"])

    with tab1:
        st.image("plot_graph/model_performance_table.png", use_container_width=True)

    with tab2:
        st.image("plot_graph/rmse_comparison.png", use_container_width=True)
        st.image("plot_graph/mae_comparison.png", use_container_width=True)
        st.image("plot_graph/mape_comparison.png", use_container_width=True)

    with tab3:
        st.image("plot_graph/R2 Score Comparison_comparison.png", use_container_width=True)
else:
    st.warning("No performance images found.")

# ======================
# FORECAST ENGINE (MULTI MODEL)
# ======================
st.subheader("🔮 Forecast Comparison")

prices = df['Price'].tolist()
lstm_series = df['LSTM_Pred'].tolist()

# storage
rf_preds, gb_preds, hybrid_preds = [], [], []

lag1 = prices[-1]
lag2 = prices[-2]

real_history = prices[-30:].copy()

for i in range(n_days):

    ma7 = np.mean(real_history[-7:])
    lstm_val = np.mean(lstm_series[-30:])

    # ======================
    # HYBRID (MAIN MODEL)
    # ======================
    if hybrid_model:
        X_hybrid = np.array([[lag1, lag2, ma7, lstm_val]])
        hybrid_pred = hybrid_model.predict(X_hybrid)[0]
    else:
        hybrid_pred = lag1  # fallback

    hybrid_preds.append(hybrid_pred)

    # ======================
    # RF SAFE PRED
    # ======================
    if rf_model:
        rf_features = list(rf_model.feature_names_in_)
        rf_input = pd.DataFrame([[lag1, lag2, ma7, lstm_val]],
                                columns=['lag1', 'lag2', 'ma7', 'LSTM_Pred'])
        rf_input = rf_input.reindex(columns=rf_features, fill_value=0)
        rf_pred = rf_model.predict(rf_input)[0]
    else:
        rf_pred = lag1

    rf_preds.append(rf_pred)

    # ======================
    # GB SAFE PRED
    # ======================
    if gb_model:
        gb_features = list(gb_model.feature_names_in_)
        gb_input = rf_input.reindex(columns=gb_features, fill_value=0)
        gb_pred = gb_model.predict(gb_input)[0]
    else:
        gb_pred = lag1

    gb_preds.append(gb_pred)

    # ======================
    # UPDATE (ANCHOR HYBRID)
    # ======================
    lag2 = lag1
    lag1 = hybrid_pred

    if i % 7 == 0:
        real_history.append(hybrid_pred)
        real_history = real_history[-30:]

# ======================
# FUTURE DATES
# ======================
future_dates = pd.date_range(
    start=df['Date'].iloc[-1] + pd.Timedelta(days=1),
    periods=n_days
)

# ======================
# PLOT
# ======================
fig, ax = plt.subplots(figsize=(14, 5))

ax.plot(future_dates, hybrid_preds, label="Hybrid (Best)", linewidth=3)
ax.plot(future_dates, rf_preds, label="Random Forest", alpha=0.7)
ax.plot(future_dates, gb_preds, label="Gradient Boosting", alpha=0.7)

ax.legend()
ax.set_title("Gold Price Forecast Comparison")

st.pyplot(fig)

# ======================
# SUMMARY TABLE
# ======================
st.subheader("📊 Forecast Summary")

summary_df = pd.DataFrame({
    "Model": ["Hybrid", "Random Forest", "Gradient Boosting"],
    "Final Price": [
        hybrid_preds[-1],
        rf_preds[-1],
        gb_preds[-1]
    ]
})

st.dataframe(summary_df, use_container_width=True)

# ======================
# GROWTH
# ======================
start_price = hybrid_preds[0]
end_price = hybrid_preds[-1]

change_pct = ((end_price - start_price) / start_price) * 100

if change_pct > 0:
    st.success(f"📈 Expected Growth: +{change_pct:.2f}%")
else:
    st.warning(f"📉 Expected Decline: {change_pct:.2f}%")
