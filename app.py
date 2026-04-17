import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from datetime import timedelta

# ======================
# 1. PAGE CONFIG
# ======================
st.set_page_config(page_title="Gold Price Prediction", layout="wide")

st.title("💰 Gold Price Prediction & Model Comparison")
st.markdown("Baseline: Random Forest | Comparing Multiple Models")

# ======================
# 2. LOAD ASSETS
# ======================
@st.cache_resource
def load_assets():
    try:
        rf = joblib.load("models/random_forest.pkl")
        lr = joblib.load("models/lstm.pkl")
        gb = joblib.load("models/gradient_boosting.pkl")
        scaler = joblib.load("models/scaler.pkl")
        metrics = joblib.load("models/metrics.pkl")
        data = pd.read_csv("Gold Price.csv")

        return rf, lr, gb, scaler, metrics, data

    except Exception as e:
        st.error("❌ Error loading models or data. Check your file paths.")
        st.stop()

rf, lr, gb, scaler, metrics, data = load_assets()

# ======================
# 3. PREP DATA
# ======================
data['Date'] = pd.to_datetime(data['Date'])
data = data.sort_values('Date')

# Use ONLY lag1 (simple + stable)
data['lag1'] = data['Close'].shift(1)
data = data.dropna()

X = data[['lag1']]
y = data['Close']

X_scaled = scaler.transform(X)

# Train-test split (same as training)
split = int(len(X_scaled) * 0.8)
X_test = X_scaled[split:]
y_test = y.iloc[split:]

# ======================
# 4. PREDICTIONS
# ======================
rf_pred = rf.predict(X_test)
lr_pred = lr.predict(X_test)
gb_pred = gb.predict(X_test)

# ======================
# 5. MODEL COMPARISON UI
# ======================
st.subheader("📊 Model Performance Comparison")

results = pd.DataFrame({
    "Model": ["Random Forest (Baseline)", "Linear Regression", "Gradient Boosting"],
    "MAE": [
        metrics["rf"]["MAE"],
        metrics["lr"]["MAE"],
        metrics["gb"]["MAE"]
    ],
    "RMSE": [
        metrics["rf"]["RMSE"],
        metrics["lr"]["RMSE"],
        metrics["gb"]["RMSE"]
    ],
    "R2": [
        metrics["rf"]["R2"],
        metrics["lr"]["R2"],
        metrics["gb"]["R2"]
    ]
})

st.dataframe(results, use_container_width=True)

# ======================
# 6. BEST MODEL
# ======================
best_model = results.sort_values("RMSE").iloc[0]

st.success(f"""
🏆 Best Model: {best_model['Model']}

RMSE: {best_model['RMSE']:.2f}  
R²: {best_model['R2']:.4f}
""")

# ======================
# 7. PLOT COMPARISON
# ======================
st.subheader("📈 Prediction Comparison")

fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(y_test.values, label="Actual", linewidth=2)
ax.plot(rf_pred, label="Random Forest")
ax.plot(lr_pred, label="Linear Regression")
ax.plot(gb_pred, label="Gradient Boosting")

ax.legend()
ax.set_title("Model Predictions vs Actual")

st.pyplot(fig)

# ======================
# 8. FORECAST SECTION
# ======================
st.subheader("🔮 Future Forecast")

years = st.slider("Years to Predict", 1, 10, 1)
days = years * 365

# Use BEST model for forecasting
model_map = {
    "Random Forest (Baseline)": rf,
    "Linear Regression": lr,
    "Gradient Boosting": gb
}

best_model_name = best_model["Model"]
model = model_map[best_model_name]

# Start from last known value
last_value = data['Close'].iloc[-1]
current_input = np.array([[last_value]])

future_preds = []

for _ in range(days):
    scaled_input = scaler.transform(current_input)
    pred = model.predict(scaled_input)[0]

    future_preds.append(pred)

    # update lag1
    current_input = np.array([[pred]])

# Create future dates
last_date = data['Date'].iloc[-1]
future_dates = [last_date + timedelta(days=i) for i in range(1, days+1)]

forecast_df = pd.DataFrame({
    "Date": future_dates,
    "Predicted Price": future_preds
})

# ======================
# 9. FORECAST PLOT
# ======================
st.subheader("📅 Forecast Plot")

fig2, ax2 = plt.subplots(figsize=(10, 5))

ax2.plot(data['Date'].tail(200), data['Close'].tail(200), label="Historical")
ax2.plot(forecast_df['Date'], forecast_df['Predicted Price'], label="Forecast")

ax2.legend()
ax2.set_title(f"Future Forecast using {best_model_name}")

st.pyplot(fig2)

# ======================
# 10. SHOW DATA
# ======================
with st.expander("📂 View Forecast Data"):
    st.dataframe(forecast_df)

# ======================
# 11. FOOTER
# ======================
st.markdown("---")
st.markdown("✅ Random Forest used as baseline. Models compared using MAE, RMSE, and R².")
