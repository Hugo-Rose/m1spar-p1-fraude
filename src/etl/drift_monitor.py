"""
Détection du drift de données par test KS (Kolmogorov-Smirnov).
Compare la distribution de 'amount' par mois vs janvier (référence).
Résultats loggés dans MLflow : experiment 'p1-data-drift'.
"""
import sys
import os

os.environ["PYSPARK_PYTHON"]        = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

_java11 = r"C:\Program Files\Eclipse Adoptium\jdk-11.0.30.7-hotspot"
if os.path.isdir(_java11):
    os.environ["JAVA_HOME"] = _java11
    os.environ["PATH"] = os.path.join(_java11, "bin") + os.pathsep + os.environ.get("PATH", "")

_winutils = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "winutils"))
if os.path.isdir(_winutils):
    os.environ["HADOOP_HOME"] = _winutils
    os.environ["PATH"] = os.path.join(_winutils, "bin") + os.pathsep + os.environ.get("PATH", "")

import mlflow
from scipy import stats
from pyspark.sql import SparkSession, functions as F


def monitor_drift_p1(spark: SparkSession,
                     silver_path: str,
                     mlflow_uri: str = "http://localhost:5000") -> None:
    """
    Teste le drift de 'amount' entre janvier (référence) et chaque
    mois suivant. p-value < 0.05 → drift significatif.

    Logue dans MLflow :
      - ks_stat        : statistique KS
      - p_value        : p-value du test
      - drift_detected : 1 si drift, 0 sinon
      - month / feature (params)
    """
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment("p1-data-drift")

    silver = spark.read.format("delta").load(silver_path)

    ref_pd = silver.filter(
        "transaction_date BETWEEN '2024-01-01' AND '2024-01-31'"
    ).select("amount").toPandas()["amount"]

    months = ["2024-02", "2024-03", "2024-04", "2024-05", "2024-06"]

    for month in months:
        curr_pd = silver.filter(
            F.col("transaction_date").startswith(month)
        ).select("amount").toPandas()["amount"]

        ks_stat, p_value = stats.ks_2samp(ref_pd, curr_pd)
        drift = int(p_value < 0.05)

        with mlflow.start_run(run_name=f"drift-amount-{month}"):
            mlflow.log_metric("ks_stat",        round(float(ks_stat), 4))
            mlflow.log_metric("p_value",         round(float(p_value), 4))
            mlflow.log_metric("drift_detected",  drift)
            mlflow.log_param("month",            month)
            mlflow.log_param("feature",          "amount")

        status = "DRIFT ⚠" if drift else "OK    ✓"
        print(f"{month} : KS={ks_stat:.4f}  p={p_value:.4f}  → {status}")


if __name__ == "__main__":
    SPARK_TMP = os.path.join(os.path.expanduser("~"), "spark_tmp")
    os.makedirs(SPARK_TMP, exist_ok=True)

    spark = SparkSession.builder \
        .appName("M1SPAR-P1-DriftMonitor") \
        .config("spark.driver.memory", "4g") \
        .config("spark.local.dir", SPARK_TMP) \
        .config("spark.sql.extensions",
                "io.delta.sql.DeltaSparkSessionExtension") \
        .config("spark.sql.catalog.spark_catalog",
                "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
        .config("spark.jars.packages",
                "io.delta:delta-spark_2.12:3.2.0") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    _root   = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    _silver = os.path.join(_root, "data", "delta", "silver", "fraud")

    monitor_drift_p1(spark, _silver)
    spark.stop()
