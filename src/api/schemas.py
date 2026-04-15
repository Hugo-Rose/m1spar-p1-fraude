"""
Schémas Pydantic — POST /predict
Validation des entrées et format de réponse.
"""
from pydantic import BaseModel, Field
from typing import Optional


class TransactionRequest(BaseModel):
    transaction_id:    str
    amount:            float = Field(gt=0.0,  lt=50_000.0)
    velocity_1h:       float = Field(ge=0.0,  le=50.0)
    night_tx_ratio:    float = Field(ge=0.0,  le=1.0)
    V14:               float
    V17:               float
    zscore_amount:     Optional[float] = 0.0
    risk_score:        Optional[float] = Field(default=0.1, ge=0.0, le=1.0)
    merchant_category: Optional[str]   = "unknown"

    model_config = {
        "json_schema_extra": {
            "example": {
                "transaction_id": "TX0000001",
                "amount": 34.14,
                "velocity_1h": 2.0,
                "night_tx_ratio": 0.08,
                "V14": 0.02,
                "V17": -0.05,
            }
        }
    }


class PredictionResponse(BaseModel):
    transaction_id:    str
    is_fraud:          bool
    fraud_probability: float = Field(ge=0.0, le=1.0)
    risk_level:        str   # LOW | MEDIUM | HIGH
    model_version:     str
    latency_ms:        float
