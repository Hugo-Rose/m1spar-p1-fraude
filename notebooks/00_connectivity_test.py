"""
Test de connectivite - M1SPAR P1 Fraude
Executer avant de commencer le TP EDA
"""
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
import time, os
from dotenv import load_dotenv

load_dotenv()

# Chemin absolu vers le dataset genere
DATASET_PATH = os.getenv(
    "DATASET_PATH",
    r"D:\m1spar-p1-fraude\data\fraud_dataset"
)

# Fix Java 17+ : Hadoop utilise Subject.getSubject() retire en Java 17
JAVA17_OPTS = (
    "--add-opens=java.base/javax.security.auth=ALL-UNNAMED "
    "--add-opens=java.base/sun.security.action=ALL-UNNAMED "
    "--add-opens=java.base/java.lang=ALL-UNNAMED "
    "--add-opens=java.base/java.util=ALL-UNNAMED "
    "--add-opens=java.base/sun.nio.ch=ALL-UNNAMED"
)

print("=" * 55)
print("  M1SPAR P1 - Test de connectivite")
print("=" * 55)

spark = SparkSession.builder \
    .appName("M1SPAR-P1-ConnectTest") \
    .config("spark.sql.adaptive.enabled", "true") \
    .config("spark.driver.memory", "4g") \
    .config("spark.driver.extraJavaOptions", JAVA17_OPTS) \
    .config("spark.executor.extraJavaOptions", JAVA17_OPTS) \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")
print(f"\nSpark version : {spark.version}")

print(f"\nChargement dataset : {DATASET_PATH}")
t0 = time.time()
df = spark.read.parquet(DATASET_PATH)
t1 = time.time()

print(f"Lignes    : {df.count():,}")
print(f"Colonnes  : {len(df.columns)}")
print(f"Temps     : {t1-t0:.2f}s")

t2 = time.time()
n = df.filter("transaction_date = '2024-03-15'").count()
print(f"Pruning   : {n:,} lignes en {time.time()-t2:.2f}s")

print("\nSchema (colonnes cles) :")
df.select("transaction_id", "amount", "country",
          "is_fraud", "transaction_date").show(3)

print("\nConnectivite OK - Pret pour le TP EDA")
spark.stop()
