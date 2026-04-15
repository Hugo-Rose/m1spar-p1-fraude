"""
Bronze ingestion : Parquet → Delta Lake Bronze
Partition par transaction_date pour le pruning.
"""
import sys
import os

# ── Fix Windows/Linux automatique ─────────────────────────────
os.environ["PYSPARK_PYTHON"]        = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

# ── Force Java 11 si disponible ───────────────────────────────
_java11 = r"C:\Program Files\Eclipse Adoptium\jdk-11.0.30.7-hotspot"
if os.path.isdir(_java11):
    os.environ["JAVA_HOME"] = _java11
    os.environ["PATH"] = os.path.join(_java11, "bin") + os.pathsep + os.environ.get("PATH", "")

# ── winutils pour Hadoop Windows ──────────────────────────────
_winutils = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "winutils"))
if os.path.isdir(_winutils):
    os.environ["HADOOP_HOME"] = _winutils
    os.environ["PATH"] = os.path.join(_winutils, "bin") + os.pathsep + os.environ.get("PATH", "")

SPARK_TMP = os.path.join(os.path.expanduser("~"), "spark_tmp")
os.makedirs(SPARK_TMP, exist_ok=True)
os.environ["SPARK_LOCAL_DIRS"] = SPARK_TMP

from pyspark.sql import SparkSession, DataFrame


def get_spark(app_name: str = "M1SPAR-P1-Bronze") -> SparkSession:
    return SparkSession.builder \
        .appName(app_name) \
        .config("spark.driver.memory", "4g") \
        .config("spark.local.dir", SPARK_TMP) \
        .config("spark.hadoop.fs.file.impl.disable.cache", "true") \
        .config("spark.sql.extensions",
                "io.delta.sql.DeltaSparkSessionExtension") \
        .config("spark.sql.catalog.spark_catalog",
                "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
        .config("spark.jars.packages",
                "io.delta:delta-spark_2.12:3.2.0") \
        .getOrCreate()


def ingest_bronze(spark: SparkSession,
                  source_path: str,
                  bronze_path: str) -> int:
    """
    Lit le dataset fraude Parquet et l'écrit en Delta Lake Bronze.
    Retourne le nombre de lignes ingérées.
    """
    df = spark.read.parquet(source_path)

    df.write \
        .format("delta") \
        .mode("overwrite") \
        .partitionBy("transaction_date") \
        .save(bronze_path)

    count = spark.read.format("delta").load(bronze_path).count()
    print(f"[Bronze] {count:,} lignes ingérées → {bronze_path}")
    return count


if __name__ == "__main__":
    _root    = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    _src     = os.path.join(_root, "data", "fraud_dataset")
    _bronze  = os.path.join(_root, "data", "delta", "bronze", "fraud")

    spark = get_spark()
    spark.sparkContext.setLogLevel("ERROR")
    ingest_bronze(spark, _src, _bronze)
    spark.stop()
