import pandas as pd
import numpy as np
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ======================
# LOAD DATA
# ======================
df = pd.read_csv("data/Gold Price.csv")
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)

# ======================
# FEATURE ENGINEERING
# ======================
# 1. Feature Engineering - BE CONSISTENT
df['Lag1'] = df['Price'].shift(1)
df['Lag2'] = df['Price'].shift(2)
df['Lag3'] = df['Price'].shift(3) # Added Lag3 to match your App
df['MA7'] = df['Price'].rolling(7).mean()
df = df.dropna()

# 2. Split
features = ['Lag1', 'Lag2', 'Lag3', 'MA7']
split = int(len(df) * 0.8)
train, test = df.iloc[:split], df.iloc[split:]

X_train = train[features]
y_train = train['Price'] # Predicting Price directly makes the App logic simpler

# 3. Model Training
rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
rf.fit(X_train, y_train)

# 4. Save both the model and a dummy scaler if you intend to use one
joblib.dump(rf, "models/random_forest.pkl")
# Note: If not using scaling for RF, remove scaler.transform from Streamlit

# ======================
# MODEL
# ======================
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train_scaled, y_train)

# ======================
# PREDICTION
# ======================
y_pred = model.predict(X_test_scaled)

# ======================
# EVALUATION
# ======================
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n===== MODEL PERFORMANCE =====")
print(f"MAE:  {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R2:   {r2:.4f}")

# ======================
# SAVE
# ======================
joblib.dump(model, "models/rf_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")

print("\n✅ Model and scaler saved!")
