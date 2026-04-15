"""
FastAPI — POST /predict
Détection de fraude temps réel < 100ms P95.
Cache Redis 1h pour les transactions déjà vues.
"""
import os
import sys
import time
import joblib
import numpy as np

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)

import redis
from fastapi import FastAPI
from src.api.schemas import TransactionRequest, PredictionResponse

app = FastAPI(
    title="M1SPAR P1 Fraud Detection API",
    version="1.0.0",
    description="Détection de fraude financière — M1SPAR J3",
)

# ── Chargement au démarrage (une seule fois) ──────────────────
_models_dir = os.path.join(_root, "models")

MODEL     = joblib.load(os.path.join(_models_dir, "best_model.pkl"))
FEATURES  = open(os.path.join(_models_dir, "features.txt")).read().strip().split("\n")
THRESHOLD = float(open(os.path.join(_models_dir, "threshold.txt")).read().strip())

try:
    r = redis.Redis(host="localhost", port=6379, db=0, decode_responses=True,
                    socket_connect_timeout=1)
    r.ping()
    _redis_ok = True
except Exception:
    r = None
    _redis_ok = False

print(f"[API] Modele charge | Features={len(FEATURES)} | Seuil={THRESHOLD} | Redis={'OK' if _redis_ok else 'OFF'}")


@app.get("/health")
async def health():
    redis_ok = False
    if r is not None:
        try:
            redis_ok = r.ping()
        except Exception:
            pass
    return {
        "status":    "ok",
        "model":     "p1-fraud-model/1",
        "threshold": THRESHOLD,
        "redis":     redis_ok,
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(tx: TransactionRequest):
    t0 = time.time()

    # 1. Cache Redis
    if r is not None:
        try:
            cached = r.get(f"pred:{tx.transaction_id}")
            if cached:
                return PredictionResponse(**eval(cached))
        except Exception:
            pass

    # 2. Vecteur de features dans le bon ordre
    feat_map = {
        "V14":                tx.V14,
        "V17":                tx.V17,
        "velocity_1h_calc":   tx.velocity_1h,
        "night_tx_ratio":     tx.night_tx_ratio,
        "zscore_amount_calc": tx.zscore_amount,
        "risk_score":         tx.risk_score,
        "amount":             tx.amount,
        "V4": 0.0, "V11": 0.0, "V12": 0.0,
    }
    X = np.array([[feat_map.get(f, 0.0) for f in FEATURES]], dtype=np.float32)

    # 3. Inférence
    proba    = float(MODEL.predict_proba(X)[0][1])
    is_fraud = proba >= THRESHOLD
    level    = "HIGH"   if proba > 0.70 \
          else "MEDIUM" if proba > 0.30 \
          else "LOW"
    latency  = round((time.time() - t0) * 1000, 1)

    result = {
        "transaction_id":    tx.transaction_id,
        "is_fraud":          is_fraud,
        "fraud_probability": round(proba, 4),
        "risk_level":        level,
        "model_version":     "p1-fraud-model/1",
        "latency_ms":        latency,
    }

    # 4. Cache Redis 1h
    if r is not None:
        try:
            r.setex(f"pred:{tx.transaction_id}", 3600, str(result))
        except Exception:
            pass

    return PredictionResponse(**result)
