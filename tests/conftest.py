"""
Fixtures partagées — crée les fichiers modèle stub pour les tests API.
Exécuté automatiquement par pytest avant la collecte des tests.
"""
import os
import joblib
import numpy as np


def pytest_configure(config):
    """Génère best_model.pkl, threshold.txt et features.txt si absents."""
    models_dir = os.path.join(os.path.dirname(__file__), "..", "models")
    os.makedirs(models_dir, exist_ok=True)

    pkl_path    = os.path.join(models_dir, "best_model.pkl")
    thresh_path = os.path.join(models_dir, "threshold.txt")
    feat_path   = os.path.join(models_dir, "features.txt")

    features = [
        "V14", "V17", "velocity_1h_calc", "night_tx_ratio",
        "zscore_amount_calc", "risk_score", "amount",
        "V4", "V11", "V12",
    ]

    if not os.path.exists(pkl_path):
        from sklearn.linear_model import LogisticRegression
        rng = np.random.default_rng(42)
        X   = rng.random((200, len(features))).astype(np.float32)
        y   = (X[:, 0] > 0.5).astype(int)
        clf = LogisticRegression(max_iter=200).fit(X, y)
        joblib.dump(clf, pkl_path)

    if not os.path.exists(thresh_path):
        with open(thresh_path, "w") as f:
            f.write("0.5")

    if not os.path.exists(feat_path):
        with open(feat_path, "w") as f:
            f.write("\n".join(features))
