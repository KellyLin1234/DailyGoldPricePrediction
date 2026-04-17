import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import timedelta

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(page_title="Gold Price 10-Year Forecaster", layout="wide")
st.title("🪙 Gold Price 10-Year Forecaster")
st.markdown("Predicting daily gold prices using Lightweight Machine Learning (Random Forest & Gradient Boosting).")

# ==========================================
# CACHED MODEL LOADING
# ==========================================
@st.cache_resource
def load_models():
    try:
        rf = joblib.load("models/random_forest_log.pkl")
        gb = joblib.load("models/gradient_boosting_log.pkl")
        return rf, gb
    except Exception as e:
        st.error(f"Error loading models. Ensure your .pkl files are in the 'models/' folder. Details: {e}")
        return None, None

rf_model, gb_model = load_models()

# ==========================================
# FAST AUTOREGRESSIVE FORECASTING ENGINE
# ==========================================
def forecast_future(model_name, initial_prices_log, steps, models):
    rf, gb = models
    
    # Use lists for fast O(1) appends during the loop
    prices_log = list(initial_prices_log)
    future_prices = []
    
    progress_bar = st.progress(0)
    
    for i in range(steps):
        # Update progress occasionally to prevent UI lag
        if i % 500 == 0:
            progress_bar.progress(i / steps)

        # 1. Feature Engineering on the fly (pulling from end of lists)
        lag1, lag2, lag3, lag5, lag10 = prices_log[-1], prices_log[-2], prices_log[-3], prices_log[-5], prices_log[-10]
        
        ma7 = np.mean(prices_log[-7:])
        ma14 = np.mean(prices_log[-14:])
        ma30 = np.mean(prices_log[-30:])
        vol7 = np.std(prices_log[-7:], ddof=1) 
        vol14 = np.std(prices_log[-14:], ddof=1)
        momentum = prices_log[-1] - prices_log[-6]
        
        # 2. Model Inference
        pred_log_return = 0.0
        
        if model_name == 'Random Forest':
            rf_feats = np.array([[lag1, lag2, lag3, lag5, lag10, ma7, ma14, vol7, vol14, momentum]])
            pred_log_return = rf.predict(rf_feats)[0]
            
        elif model_name == 'Gradient Boosting':
            gb_feats = np.array([[lag1, lag2, lag3, lag5, lag10, ma7, ma14, ma30, vol7, vol14, momentum]])
            pred_log_return = gb.predict(gb_feats)[0]
            
        # 3. State Update
        new_log_price = prices_log[-1] + pred_log_return
        prices_log.append(new_log_price)
        
        # Convert back to actual price
        future_prices.append(np.exp(new_log_price))
        
    progress_bar.empty()
    return future_prices

# ==========================================
# UI AND EXECUTION
# ==========================================
uploaded_file = st.sidebar.file_uploader("Upload Historical Gold Price CSV", type=['csv'])
forecast_years = st.sidebar.slider("Forecast Horizon (Years)", 1, 10, 10)

if uploaded_file is not None and rf_model is not None:
    # Prepare historical data
    df = pd.read_csv(uploaded_file)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    df['Price'] = pd.to_numeric(df['Price'], errors='coerce').ffill()
    
    # Recreate the scaling and log parameters
    df['Price_Log'] = np.log(df['Price'])
    df = df.dropna()
    
    if len(df) < 30:
        st.error("Dataset must contain at least 30 days of data to calculate moving averages.")
    else:
        st.success(f"Loaded data successfully up to {df['Date'].iloc[-1].strftime('%Y-%m-%d')}")
        
        if st.button("Generate Forecast 🚀"):
            # A trading year is roughly 252 days.
            steps = forecast_years * 252 
            
            # Extract the initial state needed to kickstart the autoregressive loop
            initial_prices_log = df['Price_Log'].values[-30:] # Only need 30 days for MA30
            
            last_date = df['Date'].iloc[-1]
            future_dates = [last_date + timedelta(days=i) for i in range(1, steps + 1)]
            
            models = (rf_model, gb_model)
            
            st.subheader(f"Generating Multi-Model Forecasts for the next {forecast_years} years...")
            
            with st.spinner("Simulating Random Forest..."):
                rf_preds = forecast_future('Random Forest', initial_prices_log, steps, models)
            
            with st.spinner("Simulating Gradient Boosting..."):
                gb_preds = forecast_future('Gradient Boosting', initial_prices_log, steps, models)
                
            # ==========================================
            # VISUALIZATION WITH PLOTLY
            # ==========================================
            fig = go.Figure()

            # Plot last 2 years of actual data for visual context
            historical_plot = df.iloc[-500:] 
            fig.add_trace(go.Scatter(x=historical_plot['Date'], y=historical_plot['Price'],
                                     mode='lines', name='Actual Price', line=dict(color='black', width=2)))

            # Plot future predictions
            fig.add_trace(go.Scatter(x=future_dates, y=rf_preds, mode='lines', name='Random Forest', line=dict(color='blue', dash='dot')))
            fig.add_trace(go.Scatter(x=future_dates, y=gb_preds, mode='lines', name='Gradient Boosting', line=dict(color='orange', dash='dot')))

            fig.update_layout(
                title="10-Year Autoregressive Gold Price Forecast",
                xaxis_title="Date",
                yaxis_title="Price (USD)",
                hovermode="x unified",
                template="plotly_white"
            )

            st.plotly_chart(fig, use_container_width=True)
else:
    st.info("👈 Please upload your 'Gold Price.csv' file in the sidebar to begin.")
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
