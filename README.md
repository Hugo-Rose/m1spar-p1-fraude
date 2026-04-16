# M1SPAR P1 — Détection de Fraude Financière

> **Module M1SPAR** · École IT 2025-2026  
> PySpark · Delta Lake · XGBoost · FastAPI · Docker · MLflow · Prometheus · Grafana

## Résultats obtenus

| Modèle | AUC-ROC | F1 | Précision | Rappel |
|--------|---------|-----|-----------|--------|
| **LR (meilleur)** | **1.0000** | **0.9999** | **1.0000** | **0.9999** |
| XGBoost | 1.0000 | 0.9998 | 1.0000 | 0.9997 |
| Random Forest | 1.0000 | 0.9997 | 1.0000 | 0.9993 |

> Seuil calibré : **0.849** (Recall ≥ 90%) · Dataset PaySim synthétique → séparation parfaite attendue

## Stack technique

| Composant | Technologie | Port |
|-----------|-------------|------|
| Données | Delta Lake (Bronze/Silver/Gold) | — |
| ML | XGBoost / RF / LR + MLflow | — |
| Cache | Redis 7 | 6379 |
| API | FastAPI + Uvicorn (4 workers) | 8000 |
| MLOps | MLflow SQLite | 5000 |
| Monitoring | Prometheus + Grafana | 9090 / 3000 |
| Dashboard | Streamlit + Plotly | 8501 |

---

## Lancement avec Docker (recommandé)

### Prérequis
- Docker Desktop installé et démarré
- Le modèle entraîné doit être présent dans `models/` (`best_model.pkl`, `threshold.txt`, `features.txt`)

### Démarrer tous les services

```bash
docker compose up -d
```

Cela lance **5 services** :
| Service | URL | Description |
|---------|-----|-------------|
| API FastAPI | http://localhost:8000 | Prédictions fraude |
| MLflow UI | http://localhost:5000 | Suivi des expériences ML |
| Prometheus | http://localhost:9090 | Métriques scraping |
| Grafana | http://localhost:3000 | Dashboard monitoring (admin/admin) |
| Redis | localhost:6379 | Cache des prédictions |

### Vérifier que tout tourne

```bash
# Santé de l'API
curl http://localhost:8000/health

# Test d'une prédiction fraude
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"transaction_id":"TX-001","amount":5000,"V14":-5.08,"V17":-5.23,"velocity_1h":3.0,"night_tx_ratio":0.9}'

# Voir les logs
docker compose logs api --tail=50

# Arrêter
docker compose down
```

### URLs utiles

| URL | Description |
|-----|-------------|
| http://localhost:8000/docs | Swagger UI |
| http://localhost:8000/dashboard | Dashboard temps réel |
| http://localhost:8000/stats | Stats JSON |
| http://localhost:8000/metrics | Métriques Prometheus |

---

## Lancement local (sans Docker)

### Prérequis
- Python 3.11+, Java 11+ (pour PySpark)
- `pip install -r requirements-api.txt`

### Entraîner le modèle

```bash
python src/ml/train_pipeline.py
```

### Lancer l'API

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

### Lancer le dashboard Streamlit

```bash
pip install streamlit plotly
streamlit run src/dashboard/app.py
```

### Test de charge Locust (100 utilisateurs)

```bash
pip install locust
python -m locust -f tests/load/locustfile.py --host=http://localhost:8000 \
  --users=100 --spawn-rate=10 --run-time=60s --headless \
  --html=reports/locust_j4_100users.html
```

**Résultats obtenus** : 237 req/s · p50=47ms · p95=56ms · 0.12% d'échecs

---

## Structure du projet

```
m1spar-p1-fraude/
├── src/
│   ├── etl/                  # Pipeline Delta Lake Bronze/Silver/Gold
│   ├── ml/
│   │   └── train_pipeline.py # XGBoost/RF/LR + MLflow + calibration seuil
│   ├── api/
│   │   ├── main.py           # FastAPI : /predict /stats /metrics /dashboard
│   │   └── schemas.py        # Pydantic schemas
│   └── dashboard/
│       └── app.py            # Streamlit dashboard temps réel
├── models/                   # best_model.pkl + threshold.txt + features.txt
├── monitoring/
│   ├── prometheus.yml        # Config scraping
│   └── grafana/provisioning/ # Datasource Prometheus auto-provisionnée
├── tests/
│   └── load/locustfile.py    # Test de charge Locust
├── reports/
│   └── locust_j4_100users.html  # Rapport HTML load test
├── Dockerfile                # Image API Python 3.11
├── docker-compose.yml        # 5 services
└── requirements-api.txt      # Dépendances API
```

## KPIs atteints

| KPI | Cible | Obtenu |
|-----|-------|--------|
| Précision | > 95% | **100%** |
| Rappel | > 90% | **99.9%** |
| Latence P95 | < 100ms | **56ms** |
| Throughput | 100 users | **237 req/s** |
