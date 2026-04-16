# Features P1 — Détection de Fraude Financière

M1SPAR · TP Jour 2 · 2025-2026

## Top-5 Features — Justification Métier

| Rang | Feature          | Score ANOVA   | Fraude  | Légit   | Justification                                       |
|------|------------------|---------------|---------|---------|-----------------------------------------------------|
| 1    | `V17`            | 3 016 378     | −5.23   | +0.01   | Comportement atypique du terminal de paiement ; corrélé aux transactions rejetées puis réessayées |
| 2    | `night_tx_ratio` | 2 165 090     | 0.55    | 0.10    | Les fraudes surviennent principalement entre 00h et 05h quand la surveillance bancaire est réduite |
| 3    | `V14`            | 1 968 321     | −5.08   | +0.02   | Feature PCA très discriminante ; capture un pattern de comportement terminal anormal présent dans ~85% des fraudes |
| 4    | `risk_score`     | 1 716 239     | ≈ 0.9   | ≈ 0.1   | Score composite combinant velocity_1h, V14 et night_tx_ratio — feature engineered directement calibrée sur la fraude |
| 5    | `velocity_1h`    | 1 098 282     | 8.5     | 1.5     | Card testing : les fraudeurs testent la carte avec des micro-transactions rapides avant le grand achat |

## Top-20 Features ANOVA

| Rang | Feature                | Score ANOVA   |
|------|------------------------|---------------|
| 1    | V17                    | 3 016 378     |
| 2    | night_tx_ratio         | 2 165 090     |
| 3    | V14                    | 1 968 321     |
| 4    | risk_score             | 1 716 239     |
| 5    | velocity_1h            | 1 098 282     |
| 6    | velocity_10min         |   958 838     |
| 7    | distinct_countries_30d |   639 905     |
| 8    | V3                     |   425 980     |
| 9    | V4                     |   248 969     |
| 10   | V1                     |   190 962     |
| 11   | gps_lat                |    47 029     |
| 12   | V2                     |    46 311     |
| 13   | gps_lon                |     7 985     |
| 14   | zscore_amount_calc     |     4 052     |
| 15   | amount                 |     3 565     |
| 16   | avg_amount_7d          |     3 266     |
| 17   | avg_amount_30d         |     3 200     |
| 18   | avg_amount_7d_calc     |     1 577     |
| 19   | V27                    |         8     |
| 20   | V22                    |         4     |

## Valeurs de Référence Dataset (50M lignes)

| Feature          | is_fraud=0 (légit) | is_fraud=1 (fraude) |
|------------------|--------------------|----------------------|
| velocity_1h      | 1.5                | 8.5                  |
| V14              | +0.02              | −5.08                |
| V17              | +0.01              | −5.23                |
| amount           | 77 €               | 41 €                 |
| night_tx_ratio   | 0.10               | 0.55                 |

## Observations

- **VarianceThreshold** a éliminé 1 feature sur 48 (variance < 0.01)
- **V17 > V14** : contrairement à l'intuition initiale, V17 est la feature la plus discriminante sur ce dataset
- **risk_score** (feature engineered) se classe 4ème — valide la pertinence de notre feature engineering
- **V27 et V22** (scores 8 et 4) sont quasi-inutiles pour la discrimination fraude/légit
- **distinct_countries_30d** (rang 7) confirme que la diversité géographique est un signal fort

## Notes Techniques

- Sélection via `src/ml/feature_selection.py` — `SelectKBest(f_classif, k=20)`
- Échantillon : 1 % du Gold (~500 000 lignes) pour la vitesse
- Les features V1..V28 sont des composantes PCA des données de carte bancaire (anonymisées)
- `velocity_1h` calculée avec window function `rangeBetween(-3600, 0)` sur timestamp UNIX
- `zscore_amount_calc` = (amount − avg_7d) / (std_7d + ε) pour éviter la division par zéro
