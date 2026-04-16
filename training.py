import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor

# ======================
# 1. LOAD DATA
# ======================
df = pd.read_csv("data/Gold Price.csv")

# Ensure date sorting (important for time series)
df = df.sort_values("Date")

# ======================
# 2. FEATURE ENGINEERING
# ======================
df["Close_lag1"] = df["Close"].shift(1)
df["Close_lag2"] = df["Close"].shift(2)

df = df.dropna()

features = ["Close", "Close_lag1", "Close_lag2"]

X = df[features]
y = df["Close"]

# ======================
# 3. SCALING
# ======================
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# ======================
# 4. MODEL TRAINING
# ======================
model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

model.fit(X_scaled, y)

# ======================
# 5. SAVE ARTIFACTS
# ======================
joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(features, "features.pkl")

print("✅ Training complete. Files saved.")
