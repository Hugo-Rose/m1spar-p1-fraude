"""
Tests unitaires — src/ml/feature_selection.py
Couvre : select_top_features
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

from src.ml.feature_selection import select_top_features


@pytest.fixture(scope="module")
def spark():
    from pyspark.sql import SparkSession
    s = SparkSession.builder \
        .master("local[2]") \
        .appName("test-feature-selection") \
        .config("spark.driver.memory", "2g") \
        .config("spark.local.dir", SPARK_TMP) \
        .config("spark.hadoop.fs.file.impl.disable.cache", "true") \
        .getOrCreate()
    s.range(1).count()
    return s


def _gold_df(spark, n=500):
    """Crée un Gold DataFrame minimal avec features numériques."""
    import random
    random.seed(42)
    data = [
        (
            f"TX{i:04d}",
            "2024-01-15",
            1704067200 + i,
            1 if i % 7 == 0 else 0,         # ~14% fraude
            float(i % 10),                   # V14 : varie avec i
            float((i * 3) % 10),             # V17
            float(i % 5),                    # velocity_1h
            float(i % 3) * 0.2,             # night_tx_ratio
            50.0 + (i % 20),                # amount
            float(i % 4) - 1.0,             # zscore_amount
        )
        for i in range(n)
    ]
    cols = ["transaction_id", "transaction_date", "timestamp", "is_fraud",
            "V14", "V17", "velocity_1h", "night_tx_ratio", "amount", "zscore_amount"]
    return spark.createDataFrame(data, cols)


def test_retourne_liste_features(spark):
    """select_top_features doit retourner une liste non vide."""
    df = _gold_df(spark)
    result = select_top_features(df, k=5, sample_frac=1.0)
    assert isinstance(result, list)
    assert len(result) > 0


def test_k_features_max(spark):
    """Le nombre de features retournées ne dépasse pas k."""
    df = _gold_df(spark)
    result = select_top_features(df, k=3, sample_frac=1.0)
    assert len(result) <= 3


def test_features_dans_colonnes(spark):
    """Toutes les features retournées existent dans le DataFrame."""
    df = _gold_df(spark)
    result = select_top_features(df, k=5, sample_frac=1.0)
    df_cols = set(df.columns)
    for feat in result:
        assert feat in df_cols, f"Feature '{feat}' absente du DataFrame"


def test_variance_threshold_elimine_constantes(spark):
    """Une colonne constante ne doit pas apparaître dans les features sélectionnées."""
    import random
    random.seed(42)
    # Ajout d'une colonne constante
    data = [
        (f"TX{i:04d}", "2024-01-15", 1704067200 + i,
         1 if i % 7 == 0 else 0,
         float(i % 10), float((i * 3) % 10),
         float(i % 5), float(i % 3) * 0.2,
         50.0 + (i % 20), float(i % 4) - 1.0,
         99.0)  # constante_col
        for i in range(500)
    ]
    cols = ["transaction_id", "transaction_date", "timestamp", "is_fraud",
            "V14", "V17", "velocity_1h", "night_tx_ratio", "amount",
            "zscore_amount", "constante_col"]
    df = spark.createDataFrame(data, cols)
    result = select_top_features(df, k=5, sample_frac=1.0)
    assert "constante_col" not in result
