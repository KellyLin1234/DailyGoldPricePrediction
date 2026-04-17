import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ======================
# PAGE CONFIG
# ======================
st.set_page_config(page_title="Gold Intelligence Dashboard", layout="wide")
st.title("💰 Gold Price Prediction Dashboard")

# ======================
# LOAD DATA
# ======================
df = pd.read_csv("Gold Price.csv")
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values("Date").reset_index(drop=True)

# Ensure required columns exist
if 'LSTM_Pred' not in df.columns:
    df['LSTM_Pred'] = 0

# ======================
# FEATURE ENGINEERING (MUST MATCH TRAINING)
# ======================
df['lag1'] = df['Price'].shift(1)
df['lag2'] = df['Price'].shift(2)
df['ma7'] = df['Price'].rolling(7).mean()

df = df.dropna()

features = ['lag1', 'lag2', 'ma7', 'LSTM_Pred']
X = df[features]
y = df['Price']

# ======================
# LOAD MODELS
# ======================
rf_model = joblib.load("models/random_forest.pkl")   # BASELINE
hybrid_model = joblib.load("models/hybrid_rf.pkl")   # YOUR MODEL

# ======================
# SIDEBAR
# ======================
st.sidebar.header("Forecast Settings")
years = st.sidebar.slider("Forecast Horizon (Years)", 1, 10, 5)
n_days = years * 365

# ======================
# KPI SECTION
# ======================
last_price = df['Price'].iloc[-1]

col1, col2, col3 = st.columns(3)
col1.metric("Current Gold Price", f"{last_price:,.2f}")
col2.metric("Models", "RF vs Hybrid")
col3.metric("Forecast Horizon", f"{years} Years")

# ======================
# HISTORICAL PLOT
# ======================
st.subheader("📈 Historical Gold Price")

fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(df['Date'], df['Price'], linewidth=2)
ax.set_title("Gold Price History")
st.pyplot(fig)

# ======================
# MODEL EVALUATION
# ======================
st.subheader("📊 Model Comparison")

rf_pred = rf_model.predict(X)
hybrid_pred = hybrid_model.predict(X)

def evaluate(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred)
    }

rf_metrics = evaluate(y, rf_pred)
hybrid_metrics = evaluate(y, hybrid_pred)

comparison_df = pd.DataFrame([
    ["Random Forest (Baseline)", rf_metrics["MAE"], rf_metrics["RMSE"], rf_metrics["R2"]],
    ["Hybrid Model", hybrid_metrics["MAE"], hybrid_metrics["RMSE"], hybrid_metrics["R2"]],
], columns=["Model", "MAE", "RMSE", "R²"])

st.dataframe(comparison_df, use_container_width=True)

# ======================
# VISUAL COMPARISON
# ======================
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(df['Date'], y, label="Actual", linewidth=2)
ax.plot(df['Date'], rf_pred, label="RF", alpha=0.7)
ax.plot(df['Date'], hybrid_pred, label="Hybrid", alpha=0.7)
ax.legend()
ax.set_title("Model Predictions vs Actual")
st.pyplot(fig)

# ======================
# FORECAST ENGINE (HYBRID)
# ======================
st.subheader("🔮 Forecast (Hybrid Model)")

prices = df['Price'].tolist()
lstm_series = df['LSTM_Pred'].tolist()

predictions = []

lag1 = prices[-1]
lag2 = prices[-2]
real_history = prices[-30:].copy()

for i in range(n_days):

    ma7 = np.mean(real_history[-7:])
    lstm_pred = np.mean(lstm_series[-30:])

    X_input = np.array([[lag1, lag2, ma7, lstm_pred]])
    pred = hybrid_model.predict(X_input)[0]

    predictions.append(pred)

    lag2 = lag1
    lag1 = pred

    if i % 7 == 0:
        real_history.append(pred)
        real_history = real_history[-30:]

# ======================
# FUTURE DATES
# ======================
future_dates = pd.date_range(
    start=df['Date'].iloc[-1] + pd.Timedelta(days=1),
    periods=n_days
)

forecast_df = pd.DataFrame({
    "Date": future_dates,
    "Price": predictions
})

# ======================
# FORECAST PLOT
# ======================
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(forecast_df['Date'], forecast_df['Price'], linewidth=2)
ax.set_title("Gold Price Forecast (Hybrid Model)")
st.pyplot(fig)

# ======================
# SUMMARY
# ======================
st.subheader("📊 Forecast Summary")

start_price = predictions[0]
end_price = predictions[-1]
change_pct = ((end_price - start_price) / start_price) * 100

if change_pct > 0:
    st.success(f"📈 Expected Growth: +{change_pct:.2f}% over {years} years")
else:
    st.warning(f"📉 Expected Decline: {change_pct:.2f}% over {years} years")
