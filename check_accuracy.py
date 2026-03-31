import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, confusion_matrix,
                              classification_report)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

print("Loading data and model...")
df = pd.read_csv("data/transactions.csv")

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

X = df[FEATURES].fillna(0)
y = df["is_abuse"]

scaler = joblib.load("models/scaler.pkl")
model  = joblib.load("models/xgboost_model.pkl")

X_scaled = scaler.transform(X)
_, X_test, _, y_test = train_test_split(
    X_scaled, y, test_size=0.2, stratify=y, random_state=42
)

print("Running predictions on 1.27 million test transactions...")
y_pred_proba = model.predict_proba(X_test)[:, 1]
y_pred_50    = (y_pred_proba >= 0.50).astype(int)
y_pred_85    = (y_pred_proba >= 0.85).astype(int)

print("\n" + "="*50)
print("PAYGUARD MODULE 1 — ACCURACY REPORT")
print("="*50)

print(f"\nTest set size:     {len(y_test):,} transactions")
print(f"Fraud in test:     {y_test.sum():,} ({y_test.mean()*100:.2f}%)")
print(f"Legit in test:     {(y_test==0).sum():,}")

print("\n--- At threshold 0.50 (standard) ---")
print(f"Accuracy:          {accuracy_score(y_test, y_pred_50)*100:.4f}%")
print(f"Precision:         {precision_score(y_test, y_pred_50)*100:.4f}%")
print(f"Recall:            {recall_score(y_test, y_pred_50)*100:.4f}%")
print(f"F1 Score:          {f1_score(y_test, y_pred_50)*100:.4f}%")
print(f"AUC-ROC:           {roc_auc_score(y_test, y_pred_proba)*100:.4f}%")

cm = confusion_matrix(y_test, y_pred_50)
print(f"\nConfusion Matrix:")
print(f"                 Predicted Legit  Predicted Fraud")
print(f"Actual Legit     {cm[0][0]:>10,}       {cm[0][1]:>10,}")
print(f"Actual Fraud     {cm[1][0]:>10,}       {cm[1][1]:>10,}")

tn, fp, fn, tp = cm.ravel()
print(f"\nTrue Positives  (fraud caught):        {tp:,}")
print(f"True Negatives  (legit allowed):       {tn:,}")
print(f"False Positives (legit wrongly blocked): {fp:,}")
print(f"False Negatives (fraud missed):        {fn:,}")

print("\n--- At threshold 0.85 (BLOCK decision) ---")
print(f"Accuracy:          {accuracy_score(y_test, y_pred_85)*100:.4f}%")
print(f"Precision:         {precision_score(y_test, y_pred_85)*100:.4f}%")
print(f"Recall:            {recall_score(y_test, y_pred_85)*100:.4f}%")
print(f"F1 Score:          {f1_score(y_test, y_pred_85)*100:.4f}%")

print("\n--- Full classification report ---")
print(classification_report(y_test, y_pred_50, target_names=["Legit", "Fraud"]))
print("="*50)
