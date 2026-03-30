import joblib
import numpy as np

model    = joblib.load("models/xgboost_model.pkl")
features = joblib.load("models/features.pkl")
scaler   = joblib.load("models/scaler.pkl")

test_input = np.array([[299.0, 14, 2, 1, 4, 0.01, 20.0, 0, 0.1, 720.0, 0.25, 0.06, 0.055, 0.1, 0]])
proba = model.predict_proba(test_input)[:, 1]
print(f"Test score: {proba[0]:.4f}")
print("Export not needed - model works directly with joblib")
print(f"Features ({len(features)}): {features}")
