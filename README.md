# M1SPAR P1 — Détection de Fraude Financière

> **Module M1SPAR** · École IT 2025-2026  
> Streaming temps réel · XGBoost · Kafka · Redis · MLflow

## Stack technique

| Composant | Technologie | Port |
|-----------|------------|------|
| Ingestion | Kafka 3.6 | 9092 |
| Streaming | Spark Structured Streaming 3.4 | — |
| Stockage | Delta Lake | — |
| ML | XGBoost + Isolation Forest | — |
| Cache | Redis 7.0 | 6379 |
| API | FastAPI | 8000 |
| MLOps | MLflow | 5000 |
| Monitoring | Grafana + Prometheus | 3000 / 9090 |

## KPIs cibles

| KPI | Cible |
|-----|-------|
| Précision | > 95% |
| Rappel | > 90% |
| Latence P95 | < 100ms |
| Throughput | 100K tx/s |

## Lancement rapide

```bash
# 1. Copier le fichier de config
cp .env.example .env
# REMPLIR .env avec vos chemins et credentials

# 2. Lancer l'infrastructure
docker compose up -d

# 3. Vérifier la connectivité
jupyter notebook notebooks/00_connectivity_test.ipynb

# 4. EDA
jupyter notebook notebooks/01_eda_fraude.ipynb
```

## Structure

```
m1spar-p1-fraude/
├── notebooks/          # EDA + tests connectivité
├── src/
│   ├── etl/            # Pipeline Delta Lake Bronze/Silver/Gold
│   ├── ml/             # Training XGBoost + MLflow
│   ├── api/            # FastAPI + openapi.yaml
│   └── dashboard/      # Streamlit
├── docs/               # Architecture + diagrammes
├── tests/              # pytest + Locust
└── docker/             # Docker Compose
```
