"""
Tests unitaires — gold_features.py
Couvre : compute_fraud_features, validate_features
"""
import sys
import os
import pytest

os.environ["PYSPARK_PYTHON"]        = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

_java11 = r"C:\Program Files\Eclipse Adoptium\jdk-11.0.30.7-hotspot"
if os.path.isdir(_java11):
    os.environ["JAVA_HOME"] = _java11
    os.environ["PATH"] = os.path.join(_java11, "bin") + os.pathsep + os.environ.get("PATH", "")

_winutils = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "winutils"))
if os.path.isdir(_winutils):
    os.environ["HADOOP_HOME"] = _winutils
    os.environ["PATH"] = os.path.join(_winutils, "bin") + os.pathsep + os.environ.get("PATH", "")

SPARK_TMP = os.path.join(os.path.expanduser("~"), "spark_tmp")
os.makedirs(SPARK_TMP, exist_ok=True)

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)

from src.etl.gold_features import compute_fraud_features, validate_features

_COLS = ["client_id", "transaction_id", "timestamp",
         "amount", "V14", "night_tx_ratio", "is_fraud"]


@pytest.fixture(scope="module")
def spark():
    from pyspark.sql import SparkSession
    s = SparkSession.builder \
        .master("local[2]") \
        .appName("test-gold-features") \
        .config("spark.driver.memory", "2g") \
        .config("spark.local.dir", SPARK_TMP) \
        .config("spark.hadoop.fs.file.impl.disable.cache", "true") \
        .getOrCreate()
    s.range(1).count()  # warmup JVM
    return s


def _df(spark, data):
    return spark.createDataFrame(data, _COLS)


def test_nouvelles_colonnes_ajoutees(spark):
    """compute_fraud_features doit ajouter les 5 colonnes calculées."""
    data = [
        ("C001", "TX001", 1704067200, 100.0,  0.02, 0.1, 0),
        ("C001", "TX002", 1704067500,  50.0,  0.01, 0.1, 0),
        ("C002", "TX003", 1704067200, 200.0,  0.5,  0.0, 0),
    ]
    result = compute_fraud_features(_df(spark, data))
    expected = {"velocity_1h_calc", "velocity_10min_calc",
                "avg_amount_7d_calc", "zscore_amount_calc", "risk_score"}
    assert expected.issubset(set(result.columns))


def test_velocity_1h_correct(spark):
    """velocity_1h_calc = nb de tx dans la dernière heure pour le même client."""
    # 3 transactions C001 espacées de 20 min (toutes dans la fenêtre 1h)
    data = [
        ("C001", "TX001", 1704067200, 50.0, 0.0, 0.0, 0),
        ("C001", "TX002", 1704068400, 50.0, 0.0, 0.0, 0),  # +20 min
        ("C001", "TX003", 1704069600, 50.0, 0.0, 0.0, 0),  # +40 min
        ("C002", "TX004", 1704067200, 50.0, 0.0, 0.0, 0),  # autre client
    ]
    result = compute_fraud_features(_df(spark, data))
    # TX003 voit TX001 et TX002 dans la fenêtre → velocity_1h_calc = 3
    tx3 = result.filter("transaction_id = 'TX003'").first()
    assert tx3["velocity_1h_calc"] == 3


def test_risk_score_0_9_fraud_pattern(spark):
    """risk_score = 0.9 quand velocity_1h > 5 ET V14 < -3."""
    # 7 transactions en 6 minutes → velocity_1h_calc = 7 pour la dernière
    data = [
        ("C001", f"TX{i:03d}", 1704067200 + i * 60, 50.0, -4.0, 0.1, 1)
        for i in range(7)
    ]
    result = compute_fraud_features(_df(spark, data))
    last = result.orderBy("timestamp", ascending=False).first()
    assert last["velocity_1h_calc"] >= 6
    assert last["risk_score"] == 0.9


def test_risk_score_0_6_night_pattern(spark):
    """risk_score = 0.6 quand night_tx_ratio > 0.4 (sans fraude velocity)."""
    data = [
        ("C001", "TX001", 1704067200, 50.0, 1.0, 0.6, 0),
    ]
    result = compute_fraud_features(_df(spark, data))
    row = result.first()
    assert row["risk_score"] == 0.6


def test_risk_score_0_1_normal(spark):
    """risk_score = 0.1 pour une transaction normale."""
    data = [
        ("C001", "TX001", 1704067200, 50.0, 1.0, 0.05, 0),
    ]
    result = compute_fraud_features(_df(spark, data))
    row = result.first()
    assert row["risk_score"] == 0.1


def test_zscore_amount_calc_not_null(spark):
    """zscore_amount_calc ne doit pas être null (diviseur +ε protège div/0)."""
    data = [
        ("C001", "TX001", 1704067200, 50.0, 0.0, 0.0, 0),
        ("C001", "TX002", 1704067260, 50.0, 0.0, 0.0, 0),
    ]
    result = compute_fraud_features(_df(spark, data))
    from pyspark.sql import functions as F
    null_count = result.filter(F.col("zscore_amount_calc").isNull()).count()
    assert null_count == 0


def test_validate_features_runs(spark):
    """validate_features ne doit pas lever d'exception."""
    data = [
        ("C001", "TX001", 1704067200, 100.0, -5.0, 0.6, 1),
        ("C001", "TX002", 1704067260,  50.0,  0.1, 0.1, 0),
    ]
    result = compute_fraud_features(_df(spark, data))
    # Ne doit pas lever d'exception
    validate_features(result)
