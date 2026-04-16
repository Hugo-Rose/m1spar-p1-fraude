"""
Schémas Pydantic — POST /predict
Validation des entrées et format de réponse.
"""
from pydantic import BaseModel, Field
from typing import Optional


class TransactionRequest(BaseModel):
    transaction_id:    str   = Field(..., description="Identifiant unique de la transaction")
    amount:            float = Field(..., gt=0.0, lt=50_000.0, description="Montant en euros (0–50 000)")
    velocity_1h:       float = Field(..., ge=0.0, le=50.0,     description="Nombre de transactions dans la dernière heure")
    night_tx_ratio:    float = Field(..., ge=0.0, le=1.0,      description="Proportion de transactions nocturnes (00h–05h)")
    V14:               float = Field(..., description="Composante PCA V14 (très discriminante)")
    V17:               float = Field(..., description="Composante PCA V17 (comportement terminal)")
    zscore_amount:     Optional[float] = Field(default=0.0,  description="Z-score du montant vs historique 7j")
    zscore_amount_calc:Optional[float] = Field(default=0.0,  description="Z-score calculé (window function)")
    velocity_1h_calc:  Optional[float] = Field(default=0.0,  ge=0.0, le=50.0, description="Velocity calculée (window function)")
    risk_score:        Optional[float] = Field(default=0.1,  ge=0.0, le=1.0,  description="Score de risque composite (0.1/0.6/0.9)")
    V4:                Optional[float] = Field(default=0.0,  description="Composante PCA V4")
    V11:               Optional[float] = Field(default=0.0,  description="Composante PCA V11")
    V12:               Optional[float] = Field(default=0.0,  description="Composante PCA V12")
    avg_amount_7d:     Optional[float] = Field(default=None, description="Montant moyen sur 7 jours")
    avg_amount_7d_calc:Optional[float] = Field(default=None, description="Montant moyen calculé sur 7 jours")
    merchant_category: Optional[str]   = Field(default="unknown", description="Catégorie du marchand")

    model_config = {
        "json_schema_extra": {
            "example": {
                "transaction_id": "TX-FRAUD-001",
                "amount": 150.0,
                "velocity_1h": 9.0,
                "night_tx_ratio": 0.6,
                "V14": -5.08,
                "V17": -5.23,
                "zscore_amount_calc": 3.2,
                "velocity_1h_calc": 9.0,
                "risk_score": 0.9,
                "V4": 2.1,
                "V11": -1.2,
                "V12": 0.5,
            }
        }
    }


class PredictionResponse(BaseModel):
    transaction_id:    str
    is_fraud:          bool
    fraud_probability: float = Field(ge=0.0, le=1.0, description="Probabilité de fraude (0–1)")
    risk_level:        str   = Field(description="Niveau de risque : LOW | MEDIUM | HIGH")
    model_version:     str   = Field(description="Version du modèle MLflow en production")
    latency_ms:        float = Field(description="Latence de prédiction en millisecondes")
