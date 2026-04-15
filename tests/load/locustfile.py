"""
Tests de charge Locust — API POST /predict
50 users simultanés, 60 secondes.
Cible : P95 < 300ms, 0 erreurs.

Lancement :
  locust -f tests/load/locustfile.py \
    --host http://localhost:8000 \
    --users 50 --spawn-rate 10 --run-time 60s \
    --headless --html reports/locust_report.html
"""
from locust import HttpUser, task, between
import random


class FraudAPIUser(HttpUser):
    wait_time = between(0.1, 0.5)

    @task(7)
    def predict_legit(self):
        self.client.post("/predict", json={
            "transaction_id": f"TX{random.randint(1, 10_000_000):010d}",
            "amount":         round(random.uniform(15.0, 200.0), 2),
            "velocity_1h":    round(random.uniform(1.0, 3.0), 1),
            "night_tx_ratio": round(random.uniform(0.05, 0.20), 2),
            "V14":            round(random.uniform(-0.5, 0.5), 3),
            "V17":            round(random.uniform(-0.5, 0.5), 3),
        }, name="/predict [legit]")

    @task(2)
    def predict_fraud_card_testing(self):
        self.client.post("/predict", json={
            "transaction_id": f"FX{random.randint(1, 10_000_000):010d}",
            "amount":         round(random.uniform(1.0, 50.0), 2),
            "velocity_1h":    round(random.uniform(6.0, 15.0), 1),
            "night_tx_ratio": round(random.uniform(0.45, 0.75), 2),
            "V14":            round(random.uniform(-6.0, -3.0), 3),
            "V17":            round(random.uniform(-6.0, -3.0), 3),
        }, name="/predict [fraud]")

    @task(1)
    def predict_cached(self):
        self.client.post("/predict", json={
            "transaction_id": "TX_CACHED_PERMANENT",
            "amount": 77.0, "velocity_1h": 1.5,
            "night_tx_ratio": 0.08, "V14": 0.02, "V17": 0.01,
        }, name="/predict [cache]")

    @task(1)
    def health_check(self):
        self.client.get("/health", name="/health")
