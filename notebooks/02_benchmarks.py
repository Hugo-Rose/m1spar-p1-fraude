"""
Benchmarks PySpark - M1SPAR P1 Fraude
Test 1 : Partition pruning
Test 2 : Cache vs Sans cache
Test 3 : Broadcast join
Compatible Windows + Linux/Mac
"""
import sys
import os
import time

# ── Fix Windows/Linux automatique ─────────────────────────────
os.environ["PYSPARK_PYTHON"]        = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

# ── Force Java 11 (evite conflit avec Java 17/25 system) ──────
_java11 = r"C:\Program Files\Eclipse Adoptium\jdk-11.0.30.7-hotspot"
if os.path.isdir(_java11):
    os.environ["JAVA_HOME"] = _java11
    os.environ["PATH"] = os.path.join(_java11, "bin") + os.pathsep + os.environ.get("PATH", "")

# ── winutils.exe pour Hadoop sur Windows ──────────────────────
_winutils = os.path.join(os.path.dirname(__file__), "..", "winutils")
_winutils = os.path.abspath(_winutils)
if os.path.isdir(_winutils):
    os.environ["HADOOP_HOME"] = _winutils
    os.environ["PATH"] = os.path.join(_winutils, "bin") + os.pathsep + os.environ.get("PATH", "")

SPARK_TMP = os.path.join(os.path.expanduser("~"), "spark_tmp")
os.makedirs(SPARK_TMP, exist_ok=True)
os.environ["SPARK_LOCAL_DIRS"] = SPARK_TMP

from pyspark.sql import SparkSession, functions as F
from pyspark.sql.functions import broadcast

spark = SparkSession.builder \
    .appName("M1SPAR-P1-Benchmarks") \
    .config("spark.driver.memory",                     "4g") \
    .config("spark.local.dir",                         SPARK_TMP) \
    .config("spark.hadoop.fs.file.impl.disable.cache", "true") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "fraud_dataset"))
df = spark.read.parquet(DATA_PATH)

# ── Test 1 : Full scan vs Partition pruning ───────────────────
print("=" * 55)
print("  Test 1 - Partition pruning")
print("=" * 55)

t0 = time.time()
n_full = df.filter(F.col("amount") > 0).count()
t_full = round(time.time() - t0, 2)

t0 = time.time()
n_pruning = df.filter("transaction_date = '2024-03-15'").count()
t_pruning = round(time.time() - t0, 2)

print(f"Full scan    : {n_full:,} lignes en {t_full}s")
print(f"Pruning J15  : {n_pruning:,} lignes en {t_pruning}s")
if t_pruning > 0:
    print(f"Gain x{round(t_full / t_pruning, 1)}")
# Attendu : ~52s vs ~0.38s → x137 plus rapide

# ── Test 2 : Sans cache vs Avec cache ─────────────────────────
print("\n" + "=" * 55)
print("  Test 2 - Cache")
print("=" * 55)

df_feat = df.select(
    "is_fraud", "V14", "V17",
    "velocity_1h", "amount", "night_tx_ratio"
)

# Sans cache : 2 scans
t0 = time.time()
df_feat.groupBy("is_fraud").agg(F.mean("V14")).collect()
df_feat.groupBy("is_fraud").agg(F.mean("V17")).collect()
t_no_cache = round(time.time() - t0, 2)

# Avec cache : 1 scan + RAM
df_feat.cache()
df_feat.count()  # force le chargement en memoire

t0 = time.time()
df_feat.groupBy("is_fraud").agg(F.mean("V14")).collect()
df_feat.groupBy("is_fraud").agg(F.mean("V17")).collect()
t_cache = round(time.time() - t0, 2)

print(f"Sans cache : {t_no_cache}s")
print(f"Avec cache : {t_cache}s")
if t_cache > 0:
    print(f"Gain x{round(t_no_cache / t_cache, 1)}")
# Attendu : ~99s vs ~8s → x12.4

df_feat.unpersist()

# ── Test 3 : Broadcast join ───────────────────────────────────
print("\n" + "=" * 55)
print("  Test 3 - Broadcast join")
print("=" * 55)

merchants = spark.createDataFrame([
    ("supermarche",  "alimentation", 0.02),
    ("electronique", "high-tech",    0.08),
], ["merchant_category", "sector", "base_fraud_rate"])

t0 = time.time()
df.join(merchants, "merchant_category", "left").count()
t_no_bc = round(time.time() - t0, 2)

t0 = time.time()
df.join(broadcast(merchants), "merchant_category", "left").count()
t_bc = round(time.time() - t0, 2)

print(f"Join normal    : {t_no_bc}s")
print(f"Broadcast join : {t_bc}s")
if t_bc > 0:
    print(f"Gain x{round(t_no_bc / t_bc, 1)}")

print("\n" + "=" * 55)
print("  RESUME BENCHMARKS")
print("=" * 55)
print(f"  Pruning   : x{round(t_full/t_pruning, 1) if t_pruning > 0 else 'N/A'}")
print(f"  Cache     : x{round(t_no_cache/t_cache, 1) if t_cache > 0 else 'N/A'}")
print(f"  Broadcast : x{round(t_no_bc/t_bc, 1) if t_bc > 0 else 'N/A'}")

spark.stop()
