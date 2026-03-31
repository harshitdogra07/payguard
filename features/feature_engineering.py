import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib, os

def build_features(csv_path="data/transactions.csv"):
    df = pd.read_csv(csv_path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")
    else:
        # fallback if somehow generated without date
        pass 

    # --- Derived features ---
    df["velocity_score"] = df["mandates_24h"] / (df["mandates_7d"] + 1)
    df["is_night"] = df["hour_of_day"].between(1, 5).astype(int)

    SAFE_FEATURES = [
        "amount", "hour_of_day", "day_of_week",
        "mandates_24h", "mandates_7d",
        "device_changes", "time_since_last",
        "velocity_score", "is_night"
    ]

    split_idx = int(len(df) * 0.8)

    train_df = df.iloc[:split_idx]
    test_df  = df.iloc[split_idx:]

    X_train_raw = train_df[SAFE_FEATURES]
    y_train = train_df["is_abuse"]
    
    X_test_raw = test_df[SAFE_FEATURES]
    y_test = test_df["is_abuse"]

    scaler = StandardScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train_raw), columns=SAFE_FEATURES)
    X_test = pd.DataFrame(scaler.transform(X_test_raw), columns=SAFE_FEATURES)

    os.makedirs("models", exist_ok=True)
    joblib.dump(scaler, "models/scaler_v1.pkl")
    joblib.dump(SAFE_FEATURES, "models/features_v1.pkl")

    print(f"Train: {len(X_train)} | Test: {len(X_test)}")
    print(f"Abuse in train: {y_train.sum()} ({y_train.mean()*100:.1f}%)")
    return X_train, X_test, y_train, y_test, SAFE_FEATURES

if __name__ == "__main__":
    build_features()