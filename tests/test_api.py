"""
Tests unitaires API — POST /predict
16 tests : validations Pydantic, réponses, performance.
Redis est mocké ; le modèle stub est généré par conftest.py.
"""
import os
import sys
import pytest
from unittest.mock import MagicMock, patch

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)


@pytest.fixture(scope="module")
def client():
    """Client FastAPI avec Redis mocké (pas de serveur Redis requis)."""
    mock_redis = MagicMock()
    mock_redis.get.return_value   = None
    mock_redis.ping.return_value  = True
    mock_redis.setex.return_value = True

    # Forcer le rechargement du module pour appliquer le mock Redis
    for key in list(sys.modules.keys()):
        if "src.api.main" in key:
            del sys.modules[key]

    with patch("redis.Redis", return_value=mock_redis):
        import src.api.main as api_module
        from fastapi.testclient import TestClient
        yield TestClient(api_module.app)


# ── Payload de référence ──────────────────────────────────────
TX_LEGIT = {
    "transaction_id": "TX_TEST_001",
    "amount":         77.0,
    "velocity_1h":    1.5,
    "night_tx_ratio": 0.08,
    "V14":            0.02,
    "V17":            0.01,
}


# ── Tests /health ─────────────────────────────────────────────

def test_health_returns_200(client):
    assert client.get("/health").status_code == 200


def test_health_contains_status_ok(client):
    assert client.get("/health").json()["status"] == "ok"


# ── Tests validation Pydantic (422 attendu) ───────────────────

def test_predict_amount_negative(client):
    assert client.post("/predict",
        json={**TX_LEGIT, "amount": -5.0}).status_code == 422


def test_predict_amount_zero(client):
    assert client.post("/predict",
        json={**TX_LEGIT, "amount": 0.0}).status_code == 422


def test_predict_amount_too_high(client):
    assert client.post("/predict",
        json={**TX_LEGIT, "amount": 60_000.0}).status_code == 422


def test_predict_velocity_negative(client):
    assert client.post("/predict",
        json={**TX_LEGIT, "velocity_1h": -1.0}).status_code == 422


def test_predict_velocity_too_high(client):
    assert client.post("/predict",
        json={**TX_LEGIT, "velocity_1h": 99.0}).status_code == 422


def test_predict_night_ratio_above_1(client):
    assert client.post("/predict",
        json={**TX_LEGIT, "night_tx_ratio": 1.5}).status_code == 422


def test_predict_missing_v14(client):
    payload = {k: v for k, v in TX_LEGIT.items() if k != "V14"}
    assert client.post("/predict", json=payload).status_code == 422


# ── Tests réponse valide ──────────────────────────────────────

def test_predict_returns_200_on_valid(client):
    assert client.post("/predict", json=TX_LEGIT).status_code == 200


def test_predict_response_has_all_fields(client):
    data = client.post("/predict", json=TX_LEGIT).json()
    assert {"transaction_id", "is_fraud", "fraud_probability",
            "risk_level", "model_version", "latency_ms"}.issubset(data)


def test_predict_latency_under_5000ms(client):
    assert client.post("/predict", json=TX_LEGIT).json()["latency_ms"] < 5_000


def test_predict_transaction_id_preserved(client):
    assert client.post("/predict", json=TX_LEGIT).json()[
        "transaction_id"] == TX_LEGIT["transaction_id"]


def test_predict_risk_level_valid_values(client):
    assert client.post("/predict", json=TX_LEGIT).json()[
        "risk_level"] in ("LOW", "MEDIUM", "HIGH")


def test_predict_probability_between_0_and_1(client):
    p = client.post("/predict", json=TX_LEGIT).json()["fraud_probability"]
    assert 0.0 <= p <= 1.0


def test_predict_is_fraud_is_bool(client):
    assert isinstance(
        client.post("/predict", json=TX_LEGIT).json()["is_fraud"], bool)
