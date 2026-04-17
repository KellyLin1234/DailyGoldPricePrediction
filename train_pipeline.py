import pandas as pd
import numpy as np
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# ======================
# 1. LOAD DATA
# ======================
df = pd.read_csv("data.csv")  # change to your dataset file

# ======================
# 2. FEATURE ENGINEERING
# ======================
df["Log_Return"] = np.log(df["Close"] / df["Close"].shift(1))
df = df.dropna()

features = ["Open", "High", "Low", "Volume", "Log_Return"]

X = df[features]
y = df["Close"]

# ======================
# 3. TRAIN / TEST SPLIT
# ======================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

# ======================
# 4. SCALING (IMPORTANT)
# ======================
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ======================
# 5. MODEL (BASELINE RF)
# ======================
model = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    random_state=42
)

model.fit(X_train_scaled, y_train)

# ======================
# 6. SAVE EVERYTHING (CRITICAL FIX)
# ======================
joblib.dump(model, "rf_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(features, "features.pkl")

# ======================
# 7. QUICK CHECK
# ======================
pred = model.predict(X_test_scaled)

print("Training completed successfully.")
print("Sample prediction:", pred[:5])
