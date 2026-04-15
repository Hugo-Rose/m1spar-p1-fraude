"""
Tests unitaires — data_quality.py
Couvre : run_expectations_p1
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

from src.etl.data_quality import run_expectations_p1

_COLS = ["transaction_id", "transaction_date", "amount",
         "is_fraud", "timestamp", "velocity_1h"]


@pytest.fixture(scope="module")
def spark():
    from pyspark.sql import SparkSession
    s = SparkSession.builder \
        .master("local[2]") \
        .appName("test-data-quality") \
        .config("spark.driver.memory", "2g") \
        .config("spark.local.dir", SPARK_TMP) \
        .config("spark.hadoop.fs.file.impl.disable.cache", "true") \
        .getOrCreate()
    s.range(1).count()
    return s


def _silver_df(spark, n_total=100, n_fraud=15):
    """Crée un Silver DataFrame propre avec ~15% de fraude distribuée."""
    # Fraude distribuée sur toute la plage (1 toutes les ~7 lignes)
    # pour que n'importe quel sous-ensemble garde ~15%
    data = [
        (f"TX{i:04d}", "2024-01-15", 50.0 + i * 0.1,
         1 if i % max(1, n_total // n_fraud) == 0 else 0,
         1704067200 + i,
         float(i % 5))
        for i in range(n_total)
    ]
    return spark.createDataFrame(data, _COLS)


def test_expectations_passent_sur_donnees_propres(spark):
    """run_expectations_p1 doit retourner 100% sur un Silver propre."""
    df = _silver_df(spark, n_total=100, n_fraud=15)
    result = run_expectations_p1(df, sample_size=100)
    assert result["statistics"]["success_percent"] == 100.0


def test_expectations_retourne_dict(spark):
    """run_expectations_p1 doit retourner un dictionnaire de résultats."""
    df = _silver_df(spark)
    result = run_expectations_p1(df, sample_size=50)
    assert isinstance(result, dict)
    assert "statistics" in result
    assert "results" in result


def test_expectations_echoue_si_qualite_insuffisante(spark):
    """ValueError doit être levé si le taux de succès < 90%."""
    # Données avec 80% de fraude → expectation mean hors [0.12, 0.18]
    data = [
        (f"TX{i:04d}", "2024-01-15", 50.0,
         1 if i < 80 else 0,
         1704067200 + i,
         2.0)
        for i in range(100)
    ]
    df = spark.createDataFrame(data, _COLS)
    with pytest.raises(ValueError, match="Qualité insuffisante"):
        run_expectations_p1(df, sample_size=100)
