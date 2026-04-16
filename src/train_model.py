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
df['Lag1'] = df['Price'].shift(1)
df['Lag2'] = df['Price'].shift(2)
df['MA7'] = df['Price'].rolling(7).mean()

df = df.dropna()

features = ['Lag1', 'Lag2', 'Lag3', 'MA7']
X = df[features]
y = df['Price']

# ======================
# TRAIN TEST SPLIT (NO LEAKAGE)
# ======================
split = int(len(df) * 0.8)

X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# ======================
# SCALING (ONLY X)
# ======================
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

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
