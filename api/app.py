from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import time
import uuid

from inference.predict import predict_transaction

app = FastAPI(title="PayGuard Pipeline API", version="2.0")

class TransactionInput(BaseModel):
    amount: float
    hour_of_day: int
    day_of_week: int
    mandates_24h: int
    mandates_7d: int
    device_changes: int
    time_since_last: float

@app.post("/score")
def score_transaction(txn: TransactionInput):
    t0 = time.time()
    
    velocity_score = txn.mandates_24h / (txn.mandates_7d + 1)
    is_night = 1 if 1 <= txn.hour_of_day <= 5 else 0

    features_dict = {
        "amount": txn.amount,
        "hour_of_day": txn.hour_of_day,
        "day_of_week": txn.day_of_week,
        "mandates_24h": txn.mandates_24h,
        "mandates_7d": txn.mandates_7d,
        "device_changes": txn.device_changes,
        "time_since_last": txn.time_since_last,
        "velocity_score": velocity_score,
        "is_night": is_night
    }

    txn_id = f"txn_{uuid.uuid4().hex[:8]}"
    
    try:
        result = predict_transaction(features_dict, txn_id)
        result["latency_ms"] = round((time.time() - t0) * 1000, 2)
        result["txn_id"] = txn_id
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health():
    return {"status": "ok", "version": "v1.0 (Hybrid)"}
