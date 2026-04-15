"""
Gold features : window functions comportementales P1 Fraude.

Valeurs de référence dataset :
  velocity_1h  : légit=1.5  / fraude=8.5
  V14          : légit=+0.02 / fraude=-5.08
  amount moyen : légit=77€  / fraude=41€
"""
from pyspark.sql import DataFrame, functions as F
from pyspark.sql.window import Window


def compute_fraud_features(df: DataFrame) -> DataFrame:
    """
    Calcule les features comportementales par client via window functions.
    Ajoute :
      - velocity_1h_calc     : nb tx dans la dernière heure
      - velocity_10min_calc  : nb tx dans les 10 dernières minutes
      - avg_amount_7d_calc   : montant moyen sur 7 jours
      - zscore_amount_calc   : z-score du montant vs historique 7j
      - risk_score           : score de risque métier [0.1–0.9]
    """
    # Fenêtres temporelles par client (timestamp en secondes)
    w_1h = Window.partitionBy("client_id") \
                 .orderBy("timestamp") \
                 .rangeBetween(-3600, 0)

    w_10min = Window.partitionBy("client_id") \
                    .orderBy("timestamp") \
                    .rangeBetween(-600, 0)

    w_7d = Window.partitionBy("client_id") \
                 .orderBy("timestamp") \
                 .rangeBetween(-604_800, 0)

    return df \
        .withColumn("velocity_1h_calc",
                    F.count("transaction_id").over(w_1h)) \
        .withColumn("velocity_10min_calc",
                    F.count("transaction_id").over(w_10min)) \
        .withColumn("avg_amount_7d_calc",
                    F.mean("amount").over(w_7d)) \
        .withColumn("zscore_amount_calc",
                    (F.col("amount") - F.col("avg_amount_7d_calc")) /
                    (F.stddev("amount").over(w_7d) + F.lit(1e-6))) \
        .withColumn("risk_score",
                    F.when(
                        (F.col("velocity_1h_calc") > 5) &
                        (F.col("V14") < -3.0), 0.9)
                    .when(F.col("night_tx_ratio") > 0.4, 0.6)
                    .otherwise(0.1))


def validate_features(df: DataFrame) -> None:
    """
    Valide les features calculées vs valeurs de référence.
    Attendu :
      is_fraud=0 : vel_1h≈1.5,  V14_mean≈+0.02
      is_fraud=1 : vel_1h≈8.5,  V14_mean≈-5.08
    """
    df.groupBy("is_fraud").agg(
        F.round(F.mean("velocity_1h_calc"), 2).alias("vel_1h"),
        F.round(F.mean("V14"), 3).alias("V14_mean"),
        F.round(F.mean("risk_score"), 3).alias("risk"),
    ).orderBy("is_fraud").show()


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
        .appName("M1SPAR-P1-Gold") \
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
    _gold   = os.path.join(_root, "data", "delta", "gold", "fraud")

    silver_df = spark.read.format("delta").load(_silver)
    gold_df   = compute_fraud_features(silver_df)

    gold_df.write \
        .format("delta") \
        .mode("overwrite") \
        .partitionBy("transaction_date") \
        .save(_gold)

    print(f"[Gold] {gold_df.count():,} lignes avec features")
    validate_features(gold_df)
    spark.stop()
