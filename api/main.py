from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import joblib
import time

app = FastAPI(title="PayGuard Abuse Detection API", version="1.0")

scaler    = joblib.load("models/scaler.pkl")
features  = joblib.load("models/features.pkl")
threshold = joblib.load("models/threshold.pkl")
model     = joblib.load("models/xgboost_model.pkl")

class Transaction(BaseModel):
    user_id:         int
    merchant_id:     int
    amount:          float
    hour_of_day:     int
    day_of_week:     int
    mandates_24h:    int
    mandates_7d:     int
    dispute_rate:    float
    avg_amount_diff: float
    device_changes:  int
    merchant_risk:   float
    time_since_last: float

class ScoreResponse(BaseModel):
    transaction_id: str
    abuse_score:    float
    decision:       str
    reason:         str
    latency_ms:     float

def engineer(txn: Transaction):
    velocity_score = txn.mandates_24h / (txn.mandates_7d + 1)
    amount_anomaly = abs(txn.avg_amount_diff) / (txn.amount + 1)
    composite_risk = 0.5 * txn.merchant_risk + 0.5 * txn.dispute_rate
    time_risk = (1.0 if txn.time_since_last < 1 else
                 0.8 if txn.time_since_last > 720 else 0.1)
    is_night = 1 if 1 <= txn.hour_of_day <= 5 else 0

    raw = [[
        txn.amount, txn.hour_of_day, txn.day_of_week,
        txn.mandates_24h, txn.mandates_7d, txn.dispute_rate,
        txn.avg_amount_diff, txn.device_changes, txn.merchant_risk,
        txn.time_since_last, velocity_score, amount_anomaly,
        composite_risk, time_risk, is_night
    ]]
    return scaler.transform(raw)

def explain(txn: Transaction, score: float):
    if score < 0.5:
        return "Transaction pattern looks normal"
    reasons = []
    if txn.mandates_24h > 5:
        reasons.append(f"high mandate velocity ({txn.mandates_24h} in 24h)")
    if txn.dispute_rate > 0.3:
        reasons.append(f"high dispute history ({txn.dispute_rate:.0%})")
    if abs(txn.avg_amount_diff) > 500:
        reasons.append("unusual charge amount")
    if txn.device_changes > 2:
        reasons.append("multiple device fingerprints")
    if txn.merchant_risk > 0.7:
        reasons.append("high-risk merchant")
    return "; ".join(reasons) if reasons else "anomaly pattern detected"

@app.post("/score", response_model=ScoreResponse)
async def score_transaction(txn: Transaction):
    t0 = time.time()
    try:
        X = engineer(txn)
        abuse_score = float(model.predict_proba(X)[0][1])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    if abuse_score >= 0.85:
        decision = "BLOCK"
    elif abuse_score >= 0.60:
        decision = "FLAG"
    else:
        decision = "ALLOW"

    return ScoreResponse(
        transaction_id=f"TXN-{txn.user_id}-{int(time.time())}",
        abuse_score=round(abuse_score, 4),
        decision=decision,
        reason=explain(txn, abuse_score),
        latency_ms=round((time.time() - t0) * 1000, 2)
    )

@app.get("/health")
async def health():
    return {"status": "ok", "model": "xgboost-joblib", "threshold": round(float(threshold), 3)}
