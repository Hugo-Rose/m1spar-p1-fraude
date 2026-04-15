"""
Tests unitaires ETL P1 Fraude — BLOC 2
Couvre : silver_cleaning.clean_silver
"""
import sys
import os
import pytest

# ── Fix Windows avant import PySpark ──────────────────────────
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

# Ajouter la racine du projet au path pour les imports src.*
_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)

from src.etl.silver_cleaning import clean_silver


@pytest.fixture(scope="session")
def spark():
    from pyspark.sql import SparkSession
    s = SparkSession.builder \
        .master("local[2]") \
        .appName("test-p1-etl") \
        .config("spark.driver.memory", "2g") \
        .config("spark.local.dir", SPARK_TMP) \
        .config("spark.hadoop.fs.file.impl.disable.cache", "true") \
        .getOrCreate()
    # Warmup : force la JVM à se stabiliser avant le premier test
    s.range(1).count()
    return s


# ── Schéma commun ──────────────────────────────────────────────
_COLS = ["transaction_id", "transaction_date", "amount", "is_fraud", "timestamp"]


def _df(spark, data):
    return spark.createDataFrame(data, _COLS)


# ── Tests ──────────────────────────────────────────────────────

def test_remove_negative_amounts(spark):
    """Les montants négatifs (< 0.01) doivent être supprimés."""
    data = [
        ("TX001", "2024-01-01", -5.0,  0, 1234567890),
        ("TX002", "2024-01-01",  34.0, 1, 1234567891),
        ("TX003", "2024-01-01",   0.0, 0, 1234567892),
    ]
    result = clean_silver(_df(spark, data))
    assert result.count() == 1
    assert result.first()["amount"] == 34.0


def test_remove_zero_amount(spark):
    """Un montant de 0.00 doit être supprimé (< 0.01)."""
    data = [
        ("TX001", "2024-01-01", 0.00, 0, 1234567890),
        ("TX002", "2024-01-01", 0.01, 1, 1234567891),
    ]
    result = clean_silver(_df(spark, data))
    assert result.count() == 1
    assert result.first()["transaction_id"] == "TX002"


def test_remove_amount_above_max(spark):
    """Les montants > 49 999 doivent être supprimés."""
    data = [
        ("TX001", "2024-01-01", 50_000.0, 0, 1234567890),
        ("TX002", "2024-01-01", 49_999.0, 0, 1234567891),
        ("TX003", "2024-01-01",     10.0, 1, 1234567892),
    ]
    result = clean_silver(_df(spark, data))
    assert result.count() == 2
    ids = {r["transaction_id"] for r in result.collect()}
    assert "TX001" not in ids


def test_remove_duplicates(spark):
    """Les doublons sur transaction_id doivent être supprimés (1 conservé)."""
    data = [
        ("TX001", "2024-01-01", 50.0, 0, 1234567890),
        ("TX001", "2024-01-01", 50.0, 0, 1234567890),  # doublon
        ("TX002", "2024-01-01", 75.0, 1, 1234567891),
    ]
    result = clean_silver(_df(spark, data))
    assert result.count() == 2


def test_remove_null_timestamp(spark):
    """Les lignes avec timestamp NULL doivent être supprimées."""
    data = [
        ("TX001", "2024-01-01", 20.0, 0, None),
        ("TX002", "2024-01-01", 30.0, 1, 1234567891),
    ]
    result = clean_silver(_df(spark, data))
    assert result.count() == 1
    assert result.first()["transaction_id"] == "TX002"


def test_remove_invalid_fraud_label(spark):
    """Les valeurs is_fraud hors {0,1} doivent être supprimées."""
    data = [
        ("TX001", "2024-01-01", 50.0, 2, 1234567890),   # invalide
        ("TX002", "2024-01-01", 50.0, 0, 1234567891),
        ("TX003", "2024-01-01", 50.0, 1, 1234567892),
    ]
    result = clean_silver(_df(spark, data))
    assert result.count() == 2


def test_fraud_rate_plausible(spark):
    """Le taux de fraude doit rester entre 10 % et 20 % sur données propres."""
    data = [("TX{:03d}".format(i), "2024-01-01", 50.0,
             1 if i < 15 else 0, 1234567890 + i)
            for i in range(100)]
    result = clean_silver(_df(spark, data))
    total = result.count()
    fraud  = result.filter("is_fraud = 1").count()
    rate   = fraud / total
    assert 0.10 <= rate <= 0.20, f"Taux de fraude inattendu : {rate:.2%}"


def test_clean_data_unchanged(spark):
    """Un jeu de données déjà propre doit ressortir identique."""
    data = [
        ("TX001", "2024-01-01", 100.0, 0, 1234567890),
        ("TX002", "2024-01-01",  50.0, 1, 1234567891),
        ("TX003", "2024-01-01",  25.0, 0, 1234567892),
    ]
    result = clean_silver(_df(spark, data))
    assert result.count() == 3
