# Features P1 — Détection de Fraude Financière

M1SPAR · TP Jour 2 · 2025-2026

## Top-5 Features — Justification Métier

| Rang | Feature          | Score ANOVA | Fraude  | Légit   | Justification                                       |
|------|------------------|-------------|---------|---------|-----------------------------------------------------|
| 1    | `V14`            | *(à mesurer)* | −5.08 | +0.02   | Feature PCA très discriminante ; capture un pattern de comportement terminal anormal présent dans ~85% des fraudes |
| 2    | `V17`            | *(à mesurer)* | −5.23 | +0.01   | Comportement atypique du terminal de paiement ; corrélé aux transactions rejetées puis réessayées |
| 3    | `velocity_1h`    | *(à mesurer)* |  8.5  |  1.5    | Card testing : les fraudeurs testent la carte avec des micro-transactions rapides avant le grand achat |
| 4    | `night_tx_ratio` | *(à mesurer)* | 0.55  |  0.10   | Les fraudes surviennent principalement entre 00h et 05h quand la surveillance bancaire est réduite |
| 5    | `zscore_amount`  | *(à mesurer)* | *(à mesurer)* | *(à mesurer)* | Montant anormal vs l'historique 7 jours du client : un achat 5× le panier moyen est suspect |

## Top-20 Features ANOVA (à compléter après exécution)

| Rang | Feature           | Score ANOVA |
|------|-------------------|-------------|
| 1    | V14               |             |
| 2    | V17               |             |
| 3    | velocity_1h       |             |
| 4    | night_tx_ratio    |             |
| 5    | zscore_amount     |             |
| 6    | V4                |             |
| 7    | V12               |             |
| 8    | V10               |             |
| 9    | V16               |             |
| 10   | V11               |             |
| 11   | V3                |             |
| 12   | V7                |             |
| 13   | V2                |             |
| 14   | V21               |             |
| 15   | V19               |             |
| 16   | V20               |             |
| 17   | V27               |             |
| 18   | velocity_10min    |             |
| 19   | amount            |             |
| 20   | avg_amount_7d     |             |

## Valeurs de Référence Dataset (50M lignes)

| Feature          | is_fraud=0 (légit) | is_fraud=1 (fraude) |
|------------------|--------------------|----------------------|
| velocity_1h      | 1.5                | 8.5                  |
| V14              | +0.02              | −5.08                |
| V17              | +0.01              | −5.23                |
| amount           | 77 €               | 41 €                 |
| night_tx_ratio   | 0.10               | 0.55                 |

## Notes

- Sélection via `src/ml/feature_selection.py` — `SelectKBest(f_classif, k=20)`
- Échantillon : 1 % du Gold (500 000 lignes) pour la vitesse
- Les features V1..V28 sont des composantes PCA des données de carte bancaire (anonymisées)
- `velocity_1h` calculée avec window function `rangeBetween(-3600, 0)` sur timestamp UNIX
- `zscore_amount` = (amount − avg_7d) / (std_7d + ε) pour éviter la division par zéro
