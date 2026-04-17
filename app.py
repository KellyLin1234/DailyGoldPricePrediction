import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import timedelta

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(page_title="Gold Price Predictor (Hybrid)", layout="wide")
st.title("📈 Gold Price Forecaster: Hybrid Model")
st.markdown("""
This app uses your **Hybrid Model** (Random Forest + LSTM -> Gradient Boosting Meta-Model) to autoregressively predict future prices. 
**A quick reality check:** Recursive long-term predictions (like 10 years) on financial data will compound small daily errors, often leading to exponential explosions or flatlining over thousands of days. Treat long horizons as experimental!
""")

# ==========================================
# HELPER FUNCTIONS
# ==========================================
@st.cache_resource
def load_sklearn_model(path):
    try:
        return joblib.load(path)
    except Exception as e:
        return None

@st.cache_resource
def load_keras_model(path):
    try:
        return tf.keras.models.load_model(path)
    except Exception as e:
        return None

# ==========================================
# LOAD MODELS
# ==========================================
# We need all 4 components saved in your notebook
rf_model = load_sklearn_model("models/random_forest_log.pkl")
lstm_model = load_keras_model("models/lstm_log_model.keras")
meta_model = load_sklearn_model("models/hybrid_log.pkl")
scaler = load_sklearn_model("models/lstm_log_scaler.pkl")

models_loaded = all(m is not None for m in [rf_model, lstm_model, meta_model, scaler])

# ==========================================
# SIDEBAR / INPUTS
# ==========================================
st.sidebar.header("Configuration")
data_file = st.sidebar.file_uploader("Upload Historical Gold Price CSV", type=["csv"])

# Default to 1 year as 10 years takes a while to compute step-by-step
prediction_years = st.sidebar.slider("Years to Predict", min_value=1, max_value=10, value=1)
prediction_days = prediction_years * 252 

# ==========================================
# MAIN APP LOGIC
# ==========================================
if not models_loaded:
    st.error("""
    **Missing Models!** Ensure all 4 of these files exist in your `models/` directory:
    1. `random_forest_log.pkl`
    2. `lstm_log_model.keras`
    3. `hybrid_log.pkl`
    4. `lstm_log_scaler.pkl`
    """)
elif data_file is not None:
    # 1. Load and Clean Historical Data
    df = pd.read_csv(data_file)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    df['Price'] = pd.to_numeric(df['Price'], errors='coerce').ffill()
    
    # Calculate Log Price and Log Returns (required for initial sequences)
    df['Price_Log'] = np.log(df['Price'])
    df['Log_Return'] = df['Price_Log'] - df['Price_Log'].shift(1)
    df = df.dropna().reset_index(drop=True)
    
    st.subheader("Historical Data Preview")
    st.dataframe(df.tail())

    if st.button("Generate Hybrid Forecast"):
        if len(df) < 60:
            st.error("Your uploaded data needs at least 60 days of historical data to feed the LSTM window.")
        else:
            # 2. Setup the initial arrays
            log_prices = list(df['Price_Log'].values)
            log_returns = list(df['Log_Return'].values)
            
            # Scale the historical log returns for the LSTM
            scaled_log_returns = scaler.transform(np.array(log_returns).reshape(-1, 1)).flatten().tolist()
            
            future_prices = []
            future_dates = []
            last_date = df['Date'].iloc[-1]

            # UI Progress Bar
            progress_text = "Running Hybrid Autoregressive Loop. Please wait..."
            my_bar = st.progress(0, text=progress_text)

            # 3. Autoregressive Prediction Loop
            for i in range(prediction_days):
                # --- A. RANDOM FOREST PREDICTION ---
                vol_7 = np.std(log_prices[-7:], ddof=1)
                vol_14 = np.std(log_prices[-14:], ddof=1)
                
                rf_features = [[
                    log_prices[-1], log_prices[-2], log_prices[-3], log_prices[-5], log_prices[-10],
                    np.mean(log_prices[-7:]), np.mean(log_prices[-14:]),
                    vol_7, vol_14,
                    log_prices[-1] - log_prices[-6]
                ]]
                rf_pred = rf_model.predict(rf_features)[0]
                
                # --- B. LSTM PREDICTION ---
                # Grab the last 60 scaled returns and reshape for Keras (1, 60, 1)
                lstm_input = np.array(scaled_log_returns[-60:]).reshape(1, 60, 1)
                lstm_pred_scaled = lstm_model.predict(lstm_input, verbose=0)[0][0]
                # Inverse transform back to real log return
                lstm_pred = scaler.inverse_transform([[lstm_pred_scaled]])[0][0]
                
                # --- C. META MODEL (HYBRID) PREDICTION ---
                meta_features = [[
                    rf_pred,
                    lstm_pred,
                    rf_pred - lstm_pred,
                    abs(rf_pred - lstm_pred),
                    (rf_pred + lstm_pred) / 2
                ]]
                hybrid_log_return = meta_model.predict(meta_features)[0]
                
                # --- D. UPDATE STATES FOR NEXT ITERATION ---
                next_log_price = log_prices[-1] + hybrid_log_return
                
                log_prices.append(next_log_price)
                
                # Scale the predicted log return so the LSTM can use it next step
                next_scaled_return = scaler.transform([[hybrid_log_return]])[0][0]
                scaled_log_returns.append(next_scaled_return)
                
                # Convert back to real price and store
                future_prices.append(np.exp(next_log_price))
                
                # Increment date (skipping weekends)
                last_date += timedelta(days=1)
                if last_date.weekday() >= 5: 
                    last_date += timedelta(days=2)
                future_dates.append(last_date)
                
                # Update progress bar
                my_bar.progress((i + 1) / prediction_days, text=f"Simulating Day {i+1} of {prediction_days}...")

            my_bar.empty() # Clear progress bar when done

            # 4. Compile Results
            future_df = pd.DataFrame({
                "Date": future_dates,
                "Predicted_Price": future_prices
            })

            # ==========================================
            # VISUALIZATION
            # ==========================================
            st.subheader(f"{prediction_years}-Year Hybrid Price Forecast")
            
            fig = go.Figure()
            
            # Plot last 1 year of historical data for context
            context_df = df.tail(252)
            fig.add_trace(go.Scatter(
                x=context_df['Date'], 
                y=context_df['Price'], 
                mode='lines', 
                name='Historical Price',
                line=dict(color='blue')
            ))
            
            # Plot future predictions
            fig.add_trace(go.Scatter(
                x=future_df['Date'], 
                y=future_df['Predicted_Price'], 
                mode='lines', 
                name='Hybrid Predicted Price',
                line=dict(color='purple', dash='dot')
            ))

            fig.update_layout(
                title="Gold Price: Historical vs. Hybrid Prediction",
                xaxis_title="Date",
                yaxis_title="Price",
                hovermode="x unified"
            )
            
            st.plotly_chart(fig, use_container_width=True)

            # Download Option
            csv = future_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Future Predictions as CSV",
                data=csv,
                file_name=f'hybrid_gold_predictions_{prediction_years}_years.csv',
                mime='text/csv',
            )

else:
    st.info("Please upload your historical Gold Price CSV file in the sidebar to begin.")
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
