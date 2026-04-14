"""
Test de connectivite - M1SPAR P1 Fraude
Executer avec le venv m1spar : .venv\Scripts\python.exe
"""
import os
from dotenv import load_dotenv

load_dotenv()

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
import time

DATASET_PATH = os.getenv("DATASET_PATH", "D:/m1spar-p1-fraude/data/fraud_dataset")

print("=" * 55)
print("  M1SPAR P1 - Test de connectivite")
print("=" * 55)

spark = SparkSession.builder \
    .appName("M1SPAR-P1-ConnectTest") \
    .config("spark.sql.adaptive.enabled", "true") \
    .config("spark.driver.memory", "4g") \
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
