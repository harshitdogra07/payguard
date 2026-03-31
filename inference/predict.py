import joblib
import numpy as np
import os

from utils.logger import log_decision

# Explicit loading of model version 1.0 architecture
model = joblib.load("models/xgboost_v1.pkl")
scaler = joblib.load("models/scaler_v1.pkl")
threshold = joblib.load("models/threshold_v1.pkl")

def rule_engine(txn):
    risk = 0.0
    if txn.get("mandates_24h", 0) > 10:
        risk += 0.3
    if txn.get("device_changes", 0) > 2:
        risk += 0.3
    return risk

def predict_transaction(features: dict, txn_id: str):
    # Values extracted safely mimicking exact alignment with FASTAPI dict ordering
    x = np.array([list(features.values())])
    x_scaled = scaler.transform(x)

    # ML Score execution
    ml_prob = float(model.predict_proba(x_scaled)[0][1])
    
    # Rule Score evaluation
    rule_risk = rule_engine(features)
    
    # Hybrid Combination System
    final_score = 0.7 * ml_prob + 0.3 * rule_risk

    # Smart thresholding using the explicitly tested cutoff metric
    decision = "BLOCK" if final_score >= float(threshold) else "ALLOW"

    result = {
        "score": round(final_score, 4),
        "ml_score": round(ml_prob, 4),
        "rule_risk": round(rule_risk, 4),
        "decision": decision
    }

    # Asynchronously dump to structured logs
    log_decision({
        "txn_id": txn_id,
        "score": result["score"],
        "decision": result["decision"],
        "features": features
    })
    
    return result
