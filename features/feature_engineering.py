import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib, os

def build_features(csv_path="data/transactions.csv"):
    df = pd.read_csv(csv_path)

    # --- Derived features ---
    df["velocity_score"] = df["mandates_24h"] / (df["mandates_7d"] + 1)
    df["amount_anomaly"] = df["avg_amount_diff"].abs() / (df["amount"] + 1)
    df["composite_risk"] = 0.5 * df["merchant_risk"] + 0.5 * df["dispute_rate"]
    df["time_risk"] = np.where(df["time_since_last"] < 1, 1.0,
                      np.where(df["time_since_last"] > 720, 0.8, 0.1))
    df["is_night"] = df["hour_of_day"].between(1, 5).astype(int)

    FEATURES = [
        "amount", "hour_of_day", "day_of_week",
        "mandates_24h", "mandates_7d", "dispute_rate",
        "avg_amount_diff", "device_changes", "merchant_risk",
        "time_since_last", "velocity_score", "amount_anomaly",
        "composite_risk", "time_risk", "is_night"
    ]

    X = df[FEATURES]
    y = df["is_abuse"]

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=FEATURES)

    os.makedirs("models", exist_ok=True)
    joblib.dump(scaler, "models/scaler.pkl")
    joblib.dump(FEATURES, "models/features.pkl")

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, stratify=y, random_state=42
    )

    print(f"Train: {len(X_train)} | Test: {len(X_test)}")
    print(f"Abuse in train: {y_train.sum()} ({y_train.mean()*100:.1f}%)")
    return X_train, X_test, y_train, y_test, FEATURES

if __name__ == "__main__":
    build_features()