"""
Pipeline ML P1 Fraude — M1SPAR J3
[1/6] Chargement Gold → Pandas (20%)
[2/6] Split 80/20
[3/6] Comparaison XGBoost / RF / LR → MLflow
[4/6] Courbe ROC + calibration seuil
[5/6] Feature Importance
[6/6] MLflow Registry → Production
"""
import os
import sys

# ── Fix Windows ────────────────────────────────────────────────
os.environ["PYSPARK_PYTHON"]        = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

_java11 = r"C:\Program Files\Eclipse Adoptium\jdk-11.0.30.7-hotspot"
if os.path.isdir(_java11):
    os.environ["JAVA_HOME"] = _java11
    os.environ["PATH"] = os.path.join(_java11, "bin") + os.pathsep + os.environ.get("PATH", "")

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_winutils = os.path.join(_root, "winutils")
if os.path.isdir(_winutils):
    os.environ["HADOOP_HOME"] = _winutils
    os.environ["PATH"] = os.path.join(_winutils, "bin") + os.pathsep + os.environ.get("PATH", "")

if _root not in sys.path:
    sys.path.insert(0, _root)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_curve, roc_auc_score, confusion_matrix,
    classification_report, f1_score, precision_score, recall_score,
)

SPARK_TMP = os.path.join(_root, "tmp_spark")
os.makedirs(os.path.join(_root, "reports"), exist_ok=True)
os.makedirs(os.path.join(_root, "models"),  exist_ok=True)
os.makedirs(SPARK_TMP, exist_ok=True)

GOLD_PATH    = os.path.join(_root, "data", "delta", "gold", "fraud")
REPORTS_DIR  = os.path.join(_root, "reports")
MODELS_DIR   = os.path.join(_root, "models")

TOP_FEATURES = [
    "V14", "V17", "velocity_1h_calc",
    "night_tx_ratio", "zscore_amount_calc",
    "risk_score", "amount", "V4", "V11", "V12",
]

# ── [1/6] Chargement Gold → Pandas ────────────────────────────
print("\n[1/6] Chargement Gold → Pandas (échantillon 20%)...")

from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("M1SPAR-P1-ML") \
    .config("spark.sql.extensions",
            "io.delta.sql.DeltaSparkSessionExtension") \
    .config("spark.sql.catalog.spark_catalog",
            "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
    .config("spark.jars.packages",
            "io.delta:delta-spark_2.12:3.2.0") \
    .config("spark.driver.memory",            "4g") \
    .config("spark.local.dir",                SPARK_TMP) \
    .config("spark.sql.shuffle.partitions",   "10") \
    .config("spark.hadoop.fs.file.impl.disable.cache", "true") \
    .getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

gold = spark.read.format("delta").load(GOLD_PATH)
available = [c for c in TOP_FEATURES if c in gold.columns]
print(f"      Features disponibles : {len(available)}/10  → {available}")

sample_pd = gold.select(available + ["is_fraud"]) \
                .sample(fraction=0.20, seed=42) \
                .toPandas()
spark.stop()  # libérer la JVM immédiatement

sample_pd = sample_pd.fillna(0)
print(f"      Échantillon Pandas : {len(sample_pd):,} lignes")
print(f"      Taux de fraude     : {sample_pd['is_fraud'].mean():.3f}")

X = sample_pd[available].values.astype(np.float32)
y = sample_pd["is_fraud"].values.astype(int)

# ── [2/6] Split 80/20 ─────────────────────────────────────────
print("\n[2/6] Split 80% train / 20% test...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y)
print(f"      Train : {len(X_train):,}  Test : {len(X_test):,}")

# ── [3/6] Comparaison 3 modèles ───────────────────────────────
print("\n[3/6] Comparaison XGBoost / RF / LR...")
mlflow.set_experiment("p1-model-comparison")

spw = float((y == 0).sum()) / float((y == 1).sum())
print(f"      scale_pos_weight : {spw:.2f}")

models_config = [
    ("XGBoost", xgb.XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        scale_pos_weight=spw, eval_metric="aucpr",
        random_state=42, n_jobs=-1)),
    ("RF", RandomForestClassifier(
        n_estimators=100, max_depth=8, class_weight="balanced",
        random_state=42, n_jobs=-1)),
    ("LR", Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    LogisticRegression(
            class_weight="balanced", max_iter=200, random_state=42)),
    ])),
]

best_auc, best_name, best_model = 0.0, "", None

for name, clf in models_config:
    with mlflow.start_run(run_name=name):
        clf.fit(X_train, y_train)
        y_proba = clf.predict_proba(X_test)[:, 1]
        auc     = roc_auc_score(y_test, y_proba)
        f1      = f1_score(y_test, clf.predict(X_test))
        mlflow.log_param("model",    name)
        mlflow.log_metric("auc_roc", round(auc, 4))
        mlflow.log_metric("f1",      round(f1,  4))
        mlflow.sklearn.log_model(clf, "model")
        print(f"      {name:<10s} AUC={auc:.4f}  F1={f1:.4f}")
        if auc > best_auc:
            best_auc, best_name, best_model = auc, name, clf

print(f"\n      Meilleur : {best_name} (AUC={best_auc:.4f})")

# ── [4/6] Courbe ROC + calibration seuil ──────────────────────
print("\n[4/6] Courbe ROC et calibration du seuil...")
y_proba_best = best_model.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_proba_best)
auc_score = roc_auc_score(y_test, y_proba_best)

idx_90    = np.where(tpr >= 0.90)[0][0]
THRESHOLD = float(thresholds[idx_90])
if THRESHOLD >= 1.0:
    THRESHOLD = 0.5

y_pred_cal = (y_proba_best >= THRESHOLD).astype(int)

print(f"      AUC-ROC           : {auc_score:.4f}   cible > 0.98")
print(f"      Seuil calibré     : {THRESHOLD:.3f}")
print(f"      Rappel            : {tpr[idx_90]:.3f}")
print(f"      FPR               : {fpr[idx_90]:.3f}")

cm = confusion_matrix(y_test, y_pred_cal)
tn, fp, fn, tp = cm.ravel()
f1_cal   = f1_score(y_test, y_pred_cal)
prec_cal = precision_score(y_test, y_pred_cal)
rec_cal  = recall_score(y_test, y_pred_cal)

print(f"\n      Matrice de confusion (seuil={THRESHOLD:.3f}) :")
print(f"        TN={tn:,}  FP={fp:,}  FN={fn:,}  TP={tp:,}")
print()
print(classification_report(y_test, y_pred_cal,
      target_names=["Légitimes", "Fraudes"]))
print(f"      F1-Score  : {f1_cal:.4f}   cible > 0.92")
print(f"      Précision : {prec_cal:.4f}  cible > 0.95")
print(f"      Rappel    : {rec_cal:.4f}   cible > 0.90")

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"{best_name} (AUC={auc_score:.3f})")
plt.axvline(fpr[idx_90], color="red", linestyle="--",
            label=f"Seuil={THRESHOLD:.2f} (Rappel=90%)")
plt.xlabel("Taux Faux Positifs (FPR)")
plt.ylabel("Rappel (TPR)")
plt.title("Courbe ROC — P1 Fraude M1SPAR")
plt.legend()
plt.tight_layout()
roc_path = os.path.join(REPORTS_DIR, "roc_curve.png")
plt.savefig(roc_path, dpi=150)
plt.close()
print(f"      → {roc_path}")

# ── [5/6] Feature Importance ──────────────────────────────────
print("\n[5/6] Feature Importance...")
try:
    if hasattr(best_model, "feature_importances_"):
        fi = best_model.feature_importances_
    elif hasattr(best_model, "named_steps"):
        fi = best_model.named_steps["clf"].coef_[0]
    else:
        fi = None

    if fi is not None:
        feat_imp = pd.DataFrame({
            "feature":    available[:len(fi)],
            "importance": np.abs(fi),
        }).sort_values("importance", ascending=False)

        print("      Top 5 :")
        for _, r in feat_imp.head(5).iterrows():
            print(f"        {r['feature']:<25s}  {r['importance']:.4f}")

        feat_imp.head(10).sort_values("importance").plot(
            kind="barh", x="feature", y="importance", legend=False,
            title=f"Feature Importance — {best_name} P1 Fraude",
            figsize=(8, 5))
        plt.tight_layout()
        fi_path = os.path.join(REPORTS_DIR, "feature_importance.png")
        plt.savefig(fi_path, dpi=150)
        plt.close()
        print(f"      → {fi_path}")
except Exception as e:
    print(f"      Erreur feature importance : {e}")

# ── [6/6] MLflow Registry ─────────────────────────────────────
print("\n[6/6] MLflow Registry...")
import joblib
from mlflow.tracking import MlflowClient

mlflow.set_experiment("p1-best-model")
with mlflow.start_run(run_name=f"{best_name}-best"):
    mlflow.log_params({
        "model":      best_name,
        "threshold":  round(THRESHOLD, 3),
        "train_size": len(X_train),
        "test_size":  len(X_test),
        "n_features": len(available),
        "spw":        round(spw, 2),
    })
    mlflow.log_metrics({
        "auc_roc":   round(auc_score, 4),
        "f1":        round(f1_cal,    4),
        "precision": round(prec_cal,  4),
        "recall":    round(rec_cal,   4),
        "threshold": round(THRESHOLD, 3),
    })
    if os.path.exists(roc_path):
        mlflow.log_artifact(roc_path)
    fi_path_local = os.path.join(REPORTS_DIR, "feature_importance.png")
    if os.path.exists(fi_path_local):
        mlflow.log_artifact(fi_path_local)
    mlflow.sklearn.log_model(best_model, "model")

run_id = mlflow.search_runs(
    experiment_names=["p1-best-model"],
    order_by=["start_time desc"],
).iloc[0]["run_id"]

try:
    mv = mlflow.register_model(f"runs:/{run_id}/model", "p1-fraud-model")
    MlflowClient().transition_model_version_stage(
        name="p1-fraud-model",
        version=mv.version,
        stage="Production",
    )
    print(f"      Modèle v{mv.version} → Production")
    print("      Vérifier : http://localhost:5000 → Models → p1-fraud-model")
except Exception as e:
    print(f"      Registry non disponible (MLflow server non démarré) : {e}")
    print("      Lancez : python -m mlflow server --port 5000")

# Sauvegarder localement
model_path  = os.path.join(MODELS_DIR, "best_model.pkl")
thresh_path = os.path.join(MODELS_DIR, "threshold.txt")
feat_path   = os.path.join(MODELS_DIR, "features.txt")

joblib.dump(best_model, model_path)
with open(thresh_path, "w") as f:
    f.write(str(round(THRESHOLD, 3)))
with open(feat_path, "w") as f:
    f.write("\n".join(available))

print(f"      → {model_path}")
print(f"      → {thresh_path}  ({THRESHOLD:.3f})")
print(f"      → {feat_path}")
print(f"\n✓ train_pipeline.py terminé — {best_name} AUC={auc_score:.4f}")
