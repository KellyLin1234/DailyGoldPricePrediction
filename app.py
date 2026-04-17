import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from datetime import timedelta

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(page_title="Gold Price Predictor", layout="wide")
st.title("📈 Gold Price 10-Year Forecaster")
st.markdown("""
This app uses your trained Random Forest model to autoregressively predict future gold prices.
**Note:** Long-term recursive forecasting on financial data is highly volatile due to compounding errors.
""")

# ==========================================
# HELPER FUNCTIONS
# ==========================================
@st.cache_resource
def load_model(model_path):
    try:
        return joblib.load(model_path)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def calculate_recursive_features(log_prices):
    """
    Reconstructs the exact features used in the Random Forest model:
    Lag1, Lag2, Lag3, Lag5, Lag10, MA7, MA14, Volatility7, Volatility14, Momentum
    """
    # Pandas rolling std uses ddof=1 by default
    vol_7 = np.std(log_prices[-7:], ddof=1) if len(log_prices) >= 7 else 0
    vol_14 = np.std(log_prices[-14:], ddof=1) if len(log_prices) >= 14 else 0

    features = {
        "Lag1": log_prices[-1],
        "Lag2": log_prices[-2],
        "Lag3": log_prices[-3],
        "Lag5": log_prices[-5],
        "Lag10": log_prices[-10],
        "MA7": np.mean(log_prices[-7:]),
        "MA14": np.mean(log_prices[-14:]),
        "Volatility7": vol_7,
        "Volatility14": vol_14,
        "Momentum": log_prices[-1] - log_prices[-6]  # Equivalent to current - shift(5)
    }
    
    # Return as a 2D array for sklearn prediction
    return pd.DataFrame([features])

# ==========================================
# SIDEBAR / INPUTS
# ==========================================
st.sidebar.header("Configuration")
data_file = st.sidebar.file_uploader("Upload Historical Gold Price CSV", type=["csv"])

# Default to 10 years (approx 2520 trading days)
prediction_years = st.sidebar.slider("Years to Predict", min_value=1, max_value=10, value=10)
prediction_days = prediction_years * 252 

# Load Model
# Defaulting to the Random Forest model from your notebook
model = load_model("models/random_forest_log.pkl")

# ==========================================
# MAIN APP LOGIC
# ==========================================
if data_file is not None and model is not None:
    # 1. Load and Clean Historical Data
    df = pd.read_csv(data_file)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    df['Price'] = pd.to_numeric(df['Price'], errors='coerce').ffill()
    df = df.dropna()
    
    st.subheader("Historical Data Preview")
    st.dataframe(df.tail())

    if st.button("Generate Forecast"):
        with st.spinner(f"Simulating {prediction_days} days into the future..."):
            # 2. Setup the initial window
            # We need at least the last 14 days for Volatility14, but let's grab 30 to be safe
            historical_prices = df['Price'].values
            log_prices = list(np.log(historical_prices[-30:]))
            
            future_prices = []
            last_date = df['Date'].iloc[-1]
            future_dates = []

            # 3. Autoregressive Prediction Loop
            for i in range(prediction_days):
                # Calculate features for the next step
                X_next = calculate_recursive_features(log_prices)
                
                # Predict Log Return
                pred_log_return = model.predict(X_next)[0]
                
                # Calculate new log price: Lag1 (which is log_prices[-1]) + predicted return
                next_log_price = log_prices[-1] + pred_log_return
                
                # Append to our running log_prices buffer (so the next iteration can use it)
                log_prices.append(next_log_price)
                
                # Convert back to real price and store
                next_real_price = np.exp(next_log_price)
                future_prices.append(next_real_price)
                
                # Increment date (adding business days for financial data)
                last_date += timedelta(days=1)
                # Simple logic to skip weekends
                if last_date.weekday() >= 5: 
                    last_date += timedelta(days=2)
                future_dates.append(last_date)

            # 4. Compile Results
            future_df = pd.DataFrame({
                "Date": future_dates,
                "Predicted_Price": future_prices
            })

            # ==========================================
            # VISUALIZATION
            # ==========================================
            st.subheader(f"{prediction_years}-Year Price Forecast")
            
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
                name='Predicted Price',
                line=dict(color='orange', dash='dot')
            ))

            fig.update_layout(
                title="Gold Price: Historical vs. Predicted",
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
                file_name='gold_predictions_10_years.csv',
                mime='text/csv',
            )

elif model is None:
    st.warning("Could not find the model file. Please ensure `models/random_forest_log.pkl` exists in the app directory.")
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
