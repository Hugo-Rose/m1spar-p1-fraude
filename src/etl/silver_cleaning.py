"""
Silver cleaning : Bronze → Silver avec UPSERT ACID Delta Lake.
Suppression doublons, filtrage montants impossibles, nulls timestamp.
"""
from pyspark.sql import SparkSession, DataFrame, functions as F


def clean_silver(df: DataFrame) -> DataFrame:
    """
    Nettoie le Bronze :
    - Supprime les doublons sur transaction_id
    - Filtre les montants impossibles (0.01 à 49 999)
    - Conserve uniquement is_fraud in {0, 1}
    - Supprime les lignes sans timestamp
    Retourne le DataFrame Silver propre.
    """
    return df \
        .dropDuplicates(["transaction_id"]) \
        .filter(F.col("amount").between(0.01, 49_999)) \
        .filter(F.col("is_fraud").isin([0, 1])) \
        .filter(F.col("timestamp").isNotNull())


def upsert_silver(spark: SparkSession,
                  silver_df: DataFrame,
                  silver_path: str) -> None:
    """
    Merge ACID : met à jour les existants, insère les nouveaux.
    Garantit qu'un crash ne corrompt pas le dataset.
    """
    try:
        from delta.tables import DeltaTable
        delta_available = True
    except ImportError:
        delta_available = False

    if delta_available and DeltaTable.isDeltaTable(spark, silver_path):
        DeltaTable.forPath(spark, silver_path) \
            .alias("target") \
            .merge(
                silver_df.alias("source"),
                "target.transaction_id = source.transaction_id"
            ) \
            .whenMatchedUpdateAll() \
            .whenNotMatchedInsertAll() \
            .execute()
        print("[Silver] UPSERT ACID terminé")
    else:
        silver_df.write \
            .format("delta") \
            .mode("overwrite") \
            .partitionBy("transaction_date") \
            .save(silver_path)
        print("[Silver] Première écriture terminée")


if __name__ == "__main__":
    import sys
    import os

    os.environ["PYSPARK_PYTHON"]        = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

    _java11 = r"C:\Program Files\Eclipse Adoptium\jdk-11.0.30.7-hotspot"
    if os.path.isdir(_java11):
        os.environ["JAVA_HOME"] = _java11
        os.environ["PATH"] = os.path.join(_java11, "bin") + os.pathsep + os.environ.get("PATH", "")

    _winutils = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "winutils"))
    if os.path.isdir(_winutils):
        os.environ["HADOOP_HOME"] = _winutils
        os.environ["PATH"] = os.path.join(_winutils, "bin") + os.pathsep + os.environ.get("PATH", "")

    SPARK_TMP = os.path.join(os.path.expanduser("~"), "spark_tmp")
    os.makedirs(SPARK_TMP, exist_ok=True)

    from pyspark.sql import SparkSession
    spark = SparkSession.builder \
        .appName("M1SPAR-P1-Silver") \
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
    _bronze = os.path.join(_root, "data", "delta", "bronze", "fraud")
    _silver = os.path.join(_root, "data", "delta", "silver", "fraud")

    bronze_df = spark.read.format("delta").load(_bronze)
    silver_df = clean_silver(bronze_df)
    upsert_silver(spark, silver_df, _silver)

    n = spark.read.format("delta").load(_silver).count()
    print(f"[Silver] {n:,} lignes disponibles")
    spark.stop()
