"""
Sélection automatique des top-k features par score ANOVA.
Utilise un échantillon du Gold pour limiter l'usage RAM.
"""
from __future__ import annotations
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif


def select_top_features(gold_df,
                        target_col: str = "is_fraud",
                        k: int = 20,
                        sample_frac: float = 0.01) -> list[str]:
    """
    Sélectionne les top-k features par score ANOVA F sur un échantillon.

    Étapes :
      1. Échantillonnage (1% par défaut) → Pandas
      2. VarianceThreshold : supprime les features quasi-constantes
      3. SelectKBest(f_classif) : top-k par score ANOVA

    Retourne la liste des noms de features sélectionnées.
    """
    # Colonnes non-features à exclure
    _exclude = {target_col, "transaction_id", "transaction_date",
                "timestamp", "client_id", "merchant_category",
                "merchant_id", "card_type"}

    df_pd = gold_df.sample(sample_frac).toPandas()

    feature_cols = [c for c in df_pd.columns if c not in _exclude]
    X = df_pd[feature_cols].select_dtypes(include="number")
    y = df_pd[target_col]

    # 1. Supprimer features constantes
    sel_var = VarianceThreshold(threshold=0.01)
    X_var = sel_var.fit_transform(X)
    cols_var = [X.columns[i] for i in sel_var.get_support(indices=True)]
    print(f"Après VarianceThreshold : {X.shape[1]} → {X_var.shape[1]} features")

    # 2. Top-k par ANOVA
    k_eff = min(k, X_var.shape[1])
    sel_k = SelectKBest(f_classif, k=k_eff)
    sel_k.fit(X_var, y)

    top_k = [cols_var[i] for i in sel_k.get_support(indices=True)]

    # Trier par score décroissant
    scores = {cols_var[i]: sel_k.scores_[i]
              for i in sel_k.get_support(indices=True)}
    top_k.sort(key=lambda c: scores[c], reverse=True)

    print(f"\nTop {k_eff} features (ANOVA F-score) :")
    for i, feat in enumerate(top_k, 1):
        print(f"  {i:2d}. {feat:<30s}  score={scores[feat]:.1f}")

    return top_k
