import requests

BASE = "http://localhost:8000"

normal = {
    "user_id": 1234, "merchant_id": 500, "amount": 299.0,
    "hour_of_day": 14, "day_of_week": 2,
    "mandates_24h": 1, "mandates_7d": 4,
    "dispute_rate": 0.01, "avg_amount_diff": 20.0,
    "device_changes": 0, "merchant_risk": 0.1,
    "time_since_last": 720.0
}

abusive = {
    "user_id": 9999, "merchant_id": 888, "amount": 5000.0,
    "hour_of_day": 3, "day_of_week": 6,
    "mandates_24h": 15, "mandates_7d": 40,
    "dispute_rate": 0.75, "avg_amount_diff": 4500.0,
    "device_changes": 8, "merchant_risk": 0.9,
    "time_since_last": 0.2
}

for label, txn in [("Normal", normal), ("Abusive", abusive)]:
    r = requests.post(f"{BASE}/score", json=txn)
    res = r.json()
    print(f"\n--- {label} Transaction ---")
    print(f"  Score:    {res['abuse_score']}")
    print(f"  Decision: {res['decision']}")
    print(f"  Reason:   {res['reason']}")
    print(f"  Latency:  {res['latency_ms']}ms")
