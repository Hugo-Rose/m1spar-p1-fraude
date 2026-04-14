# Architecture Technique — P1 Fraude Financière
**M1SPAR · 2025-2026**

---

## 1. Vue d'ensemble

```
┌─────────────────────────────────────────────────────────────────┐
│                    ARCHITECTURE P1 — FRAUDE                     │
│                    Streaming temps réel                         │
└─────────────────────────────────────────────────────────────────┘

[Source]          [Ingest]         [Store]          [Serve]
Transactions ──▶  Kafka        ──▶ Delta Lake   ──▶ XGBoost
(POS/Web/App)    topic:fraud       Bronze/Silver    Model
                    │                  │               │
                    ▼                  ▼               ▼
               Structured         Feature Store   FastAPI
               Streaming          (Redis)         /predict
               (Spark 3.4)             │         (<100ms)
                    │                  ▼               │
                    ▼              MLflow          Grafana
               Data Quality       Registry        Dashboard
               (Great Exp.)       (Staging →      + Alertes
                                  Production)
```

---

## 2. Justification des choix technologiques

| Composant | Choix retenu | Alternative rejetée | Justification |
|-----------|-------------|---------------------|---------------|
| **Ingestion** | Kafka 3.6 | RabbitMQ | Throughput 100K tx/s, replay, partitioning par clé |
| **Processing** | Spark Structured Streaming | Flink | Intégration MLlib, familiarité équipe, écosystème Python |
| **Stockage** | Delta Lake | Parquet brut | ACID, time travel, schema evolution, Z-ordering |
| **ML** | XGBoost + Isolation Forest | Random Forest | Gestion déséquilibre classes, vitesse d'inférence <20ms |
| **Cache** | Redis 7.0 | Memcached | Structures de données riches, TTL configurable, pipeline |
| **API** | FastAPI | Flask | Async natif, OpenAPI auto-généré, performances x4 |
| **MLOps** | MLflow | W&B | Open-source, self-hosted, model registry intégré |
| **Monitoring** | Grafana + Prometheus | Datadog | Gratuit, intégration Kafka/Spark native |

---

## 3. Décision Streaming vs Batch

**DÉCISION : STREAMING OBLIGATOIRE**

| Critère | Valeur | Décision |
|---------|--------|----------|
| Latence cible | < 100ms | → Streaming (batch impossible) |
| Volume | 100K tx/s | → Kafka obligatoire |
| Fraude en temps réel | Oui | → Décision immédiate requise |

### Pipeline Streaming — Décomposition latence

```
Transaction reçue (Kafka topic "transactions")    0ms
Spark Structured Streaming consomme             +10ms
Feature extraction depuis Redis cache           + 5ms
XGBoost inference (modèle chargé en mémoire)   +20ms
Score combiné IF + XGBoost                      + 3ms
Décision fraud/legit envoyée (Kafka/API)        + 5ms
                                               ─────
Total estimé                                   ~43ms  << 100ms ✅
```

### Feature Engineering en Streaming

| Feature | Window | Calcul |
|---------|--------|--------|
| `velocity_1h` | Sliding 1h | COUNT par user |
| `velocity_10min` | Sliding 10min | COUNT par user |
| `distinct_merchants_7d` | Tumbling 7d | COUNT DISTINCT |
| `zscore_amount` | Global Rolling | (amount - mean) / std |
| `night_tx_ratio` | Tumbling 30d | COUNT(night) / COUNT(total) |

---

## 4. Estimation des ressources cluster

### Développement (J1–J3)

| Ressource | Config | Coût/h |
|-----------|--------|--------|
| Driver | Standard_DS3_v2 · 14 GB · 4 vCPU | — |
| Workers | 2× Standard_DS3_v2 | — |
| Total estimé | — | ~0.50 $/h |

### Production (J4–J5, tests de charge)

| Ressource | Config | Justification |
|-----------|--------|---------------|
| Driver | Standard_DS4_v2 · 28 GB · 8 vCPU | Training XGBoost complet |
| Workers | 4× Standard_DS3_v2 (auto-scale 2→8) | 100K tx/s throughput |
| Kafka | 3 brokers m5.large | Réplication facteur 3 |
| Redis | r6g.large · 13 GB RAM | Feature store + cache |
| Total estimé | — | ~2.50 $/h |

**Justification mémoire Spark :**
- Dataset 10 GB Parquet → ~30 GB décompressé en mémoire
- Règle : 3-4× taille données en RAM driver
- EDA par chunks : 4 GB suffisant
- Training XGBoost complet : 28 GB nécessaire

---

## 5. Schéma de données

### Flux Bronze → Silver → Gold (Delta Lake)

```
Bronze (raw)          Silver (cleaned)        Gold (features)
─────────────────     ─────────────────────   ──────────────────────
transaction_id        transaction_id          transaction_id
transaction_date  ──▶ transaction_date    ──▶ is_fraud (target)
amount                amount_eur              V1…V28 (PCA)
country               country_iso             velocity_1h
merchant_id           merchant_category       velocity_10min
is_fraud              is_fraud                night_tx_ratio
V1…V28                V1…V28 (validated)      zscore_amount
(62 colonnes)         (nulls supprimés)       tx_count_7d
                                              distinct_countries_30d
                                              (11 features ML)
```

---

## 6. Définition des KPIs

| KPI | Cible | Outil |
|-----|-------|-------|
| Précision (Precision) | > 95% | MLflow |
| Rappel (Recall) | > 90% | MLflow |
| F1-Score | > 0.92 | MLflow |
| AUC-ROC | > 0.98 | MLflow |
| Latence API P95 | < 100ms | Locust |
| Throughput | 100K tx/s | Kafka metrics |
| Coverage tests | > 80% | pytest-cov |
| Uptime J5 | 100% | Grafana |

---

## 7. Sécurité et contraintes

- Credentials : jamais dans Git (`.env` gitignored)
- HTTPS obligatoire en production
- Authentification API : Bearer Token
- PII : pas de données personnelles nominatives dans les logs
- Audit trail : toutes les décisions loggées avec transaction_id
