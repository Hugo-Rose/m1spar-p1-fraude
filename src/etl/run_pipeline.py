"""
Orchestrateur Bronze → Silver → Gold.
Lance les 3 étapes dans l'ordre avec gestion d'erreurs.
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

SPARK_TMP = os.path.join(os.path.expanduser("~"), "spark_tmp")
os.makedirs(SPARK_TMP, exist_ok=True)

from pyspark.sql import SparkSession
from src.etl.bronze_ingestion import ingest_bronze
from src.etl.silver_cleaning import clean_silver, upsert_silver
from src.etl.gold_features import compute_fraud_features

_root   = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC     = os.path.join(_root, "data", "fraud_dataset")
BRONZE  = os.path.join(_root, "data", "delta", "bronze", "fraud")
SILVER  = os.path.join(_root, "data", "delta", "silver", "fraud")
GOLD    = os.path.join(_root, "data", "delta", "gold",   "fraud")


def run():
    spark = SparkSession.builder \
        .appName("M1SPAR-P1-Pipeline") \
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
    spark.sparkContext.setLogLevel("ERROR")

    print("\n[Pipeline] ── Étape 1 : Bronze ─────────────────────")
    ingest_bronze(spark, SRC, BRONZE)

    print("\n[Pipeline] ── Étape 2 : Silver ─────────────────────")
    bronze_df = spark.read.format("delta").load(BRONZE)
    silver_df = clean_silver(bronze_df)
    upsert_silver(spark, silver_df, SILVER)

    print("\n[Pipeline] ── Étape 3 : Gold ───────────────────────")
    silver_df2 = spark.read.format("delta").load(SILVER)
    gold_df    = compute_fraud_features(silver_df2)
    gold_df.write \
        .format("delta") \
        .mode("overwrite") \
        .partitionBy("transaction_date") \
        .save(GOLD)
    print(f"[Gold] {gold_df.count():,} lignes écrites")

    print("\n[Pipeline] ── TERMINÉ ───────────────────────────────")
    spark.stop()


if __name__ == "__main__":
    run()
