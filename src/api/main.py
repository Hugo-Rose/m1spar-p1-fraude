"""
FastAPI — POST /predict
Détection de fraude temps réel < 100ms P95.
Cache Redis 1h pour les transactions déjà vues.
Dashboard temps réel : GET /dashboard
"""
import os
import sys
import time
import joblib
import numpy as np
from collections import deque

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)

import redis
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, PlainTextResponse
from src.api.schemas import TransactionRequest, PredictionResponse

app = FastAPI(
    title="M1SPAR P1 Fraud Detection API",
    version="1.0.0",
    description="Détection de fraude financière — M1SPAR J3",
)

# ── Chargement au démarrage ───────────────────────────────────
_models_dir = os.path.join(_root, "models")

MODEL     = joblib.load(os.path.join(_models_dir, "best_model.pkl"))
FEATURES  = open(os.path.join(_models_dir, "features.txt")).read().strip().split("\n")
THRESHOLD = float(open(os.path.join(_models_dir, "threshold.txt")).read().strip())

try:
    r = redis.Redis(host="localhost", port=6379, db=0, decode_responses=True,
                    socket_connect_timeout=1)
    r.ping()
    _redis_ok = True
except Exception:
    r = None
    _redis_ok = False

print(f"[API] Modele charge | Features={len(FEATURES)} | Seuil={THRESHOLD} | Redis={'OK' if _redis_ok else 'OFF'}")

# ── Stats en mémoire ─────────────────────────────────────────
_stats = {
    "total": 0,
    "fraud": 0,
    "legit": 0,
    "latency_sum": 0.0,
    "start_time": time.time(),
}
_recent = deque(maxlen=50)  # 50 dernières transactions


# ── Endpoints ────────────────────────────────────────────────
@app.get("/health")
async def health():
    redis_ok = False
    if r is not None:
        try:
            redis_ok = r.ping()
        except Exception:
            pass
    return {
        "status":    "ok",
        "model":     "p1-fraud-model/1",
        "threshold": THRESHOLD,
        "redis":     redis_ok,
    }


@app.get("/stats")
async def stats():
    total = _stats["total"]
    fraud_rate = round(_stats["fraud"] / total * 100, 2) if total > 0 else 0.0
    avg_latency = round(_stats["latency_sum"] / total, 1) if total > 0 else 0.0
    uptime = round(time.time() - _stats["start_time"])
    return {
        "total_predictions": total,
        "fraud_count":       _stats["fraud"],
        "legit_count":       _stats["legit"],
        "fraud_rate_pct":    fraud_rate,
        "avg_latency_ms":    avg_latency,
        "uptime_seconds":    uptime,
        "recent":            list(_recent),
    }


@app.get("/metrics", response_class=PlainTextResponse)
async def metrics():
    total = _stats["total"]
    fraud = _stats["fraud"]
    legit = _stats["legit"]
    avg_lat = round(_stats["latency_sum"] / total, 1) if total > 0 else 0.0
    fraud_rate = round(fraud / total * 100, 2) if total > 0 else 0.0
    return (
        "# HELP fraud_predictions_total Total predictions\n"
        "# TYPE fraud_predictions_total counter\n"
        f"fraud_predictions_total {total}\n"
        "# HELP fraud_detected_total Fraud detections\n"
        "# TYPE fraud_detected_total counter\n"
        f"fraud_detected_total {fraud}\n"
        "# HELP fraud_rate_percent Current fraud rate\n"
        "# TYPE fraud_rate_percent gauge\n"
        f"fraud_rate_percent {fraud_rate}\n"
        "# HELP predict_latency_ms_avg Average prediction latency\n"
        "# TYPE predict_latency_ms_avg gauge\n"
        f"predict_latency_ms_avg {avg_lat}\n"
    )


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    return HTMLResponse(content="""<!DOCTYPE html>
<html lang="fr">
<head>
  <meta charset="UTF-8">
  <title>M1SPAR — Fraud Detection Dashboard</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: 'Segoe UI', sans-serif; background: #0f172a; color: #e2e8f0; padding: 24px; }
    h1 { font-size: 1.5rem; color: #38bdf8; margin-bottom: 20px; }
    .cards { display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin-bottom: 24px; }
    .card { background: #1e293b; border-radius: 12px; padding: 20px; }
    .card .label { font-size: 0.75rem; color: #94a3b8; text-transform: uppercase; letter-spacing: .05em; }
    .card .value { font-size: 2rem; font-weight: 700; margin-top: 6px; }
    .fraud  { color: #f87171; }
    .legit  { color: #4ade80; }
    .total  { color: #38bdf8; }
    .latency{ color: #fb923c; }
    .charts { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 24px; }
    .chart-box { background: #1e293b; border-radius: 12px; padding: 20px; }
    .chart-box h2 { font-size: 0.85rem; color: #94a3b8; margin-bottom: 12px; text-transform: uppercase; }
    table { width: 100%; border-collapse: collapse; background: #1e293b; border-radius: 12px; overflow: hidden; }
    th { background: #334155; padding: 10px 14px; font-size: 0.75rem; text-transform: uppercase; color: #94a3b8; text-align: left; }
    td { padding: 9px 14px; font-size: 0.82rem; border-bottom: 1px solid #334155; }
    .badge { padding: 2px 8px; border-radius: 999px; font-size: 0.72rem; font-weight: 600; }
    .badge-fraud { background: #450a0a; color: #f87171; }
    .badge-legit { background: #052e16; color: #4ade80; }
    .badge-high   { background: #450a0a; color: #f87171; }
    .badge-medium { background: #431407; color: #fb923c; }
    .badge-low    { background: #052e16; color: #4ade80; }
    .dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; background: #4ade80;
           animation: pulse 1.5s infinite; margin-right: 6px; }
    @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.3} }
  </style>
</head>
<body>
  <h1><span class="dot"></span>M1SPAR P1 — Fraud Detection Dashboard</h1>

  <div class="cards">
    <div class="card"><div class="label">Total prédictions</div><div class="value total" id="total">0</div></div>
    <div class="card"><div class="label">Fraudes détectées</div><div class="value fraud" id="fraud">0</div></div>
    <div class="card"><div class="label">Taux de fraude</div><div class="value fraud" id="rate">0.00%</div></div>
    <div class="card"><div class="label">Latence moyenne</div><div class="value latency" id="latency">0 ms</div></div>
  </div>

  <div class="charts">
    <div class="chart-box">
      <h2>Fraude vs Légitime</h2>
      <canvas id="donut" height="180"></canvas>
    </div>
    <div class="chart-box">
      <h2>Taux de fraude (dernières 20 req.)</h2>
      <canvas id="line" height="180"></canvas>
    </div>
  </div>

  <table>
    <thead>
      <tr>
        <th>Transaction ID</th>
        <th>Probabilité</th>
        <th>Résultat</th>
        <th>Niveau</th>
        <th>Latence</th>
      </tr>
    </thead>
    <tbody id="tbody"></tbody>
  </table>

<script>
const donutCtx = document.getElementById('donut').getContext('2d');
const donut = new Chart(donutCtx, {
  type: 'doughnut',
  data: { labels: ['Légitime','Fraude'], datasets: [{ data:[1,0], backgroundColor:['#4ade80','#f87171'], borderWidth:0 }] },
  options: { plugins: { legend: { labels: { color:'#e2e8f0' } } }, cutout:'65%' }
});

const lineCtx = document.getElementById('line').getContext('2d');
const lineData = { labels: [], datasets: [{ label:'Taux fraude %', data:[], borderColor:'#f87171', backgroundColor:'rgba(248,113,113,0.1)', fill:true, tension:0.4 }] };
const lineChart = new Chart(lineCtx, {
  type: 'line',
  data: lineData,
  options: { scales: { x:{ ticks:{color:'#94a3b8'}, grid:{color:'#334155'} }, y:{ ticks:{color:'#94a3b8'}, grid:{color:'#334155'}, min:0, max:100 } }, plugins:{ legend:{ labels:{color:'#e2e8f0'} } } }
});

let prevTotal = 0;
const fraudWindow = [];

async function refresh() {
  try {
    const res = await fetch('/stats');
    const d = await res.json();

    document.getElementById('total').textContent   = d.total_predictions;
    document.getElementById('fraud').textContent   = d.fraud_count;
    document.getElementById('rate').textContent    = d.fraud_rate_pct + '%';
    document.getElementById('latency').textContent = d.avg_latency_ms + ' ms';

    donut.data.datasets[0].data = [d.legit_count, d.fraud_count];
    donut.update();

    if (d.total_predictions !== prevTotal) {
      prevTotal = d.total_predictions;
      fraudWindow.push(d.fraud_rate_pct);
      if (fraudWindow.length > 20) fraudWindow.shift();
      lineData.labels = fraudWindow.map((_,i) => i+1);
      lineData.datasets[0].data = [...fraudWindow];
      lineChart.update();
    }

    const tbody = document.getElementById('tbody');
    tbody.innerHTML = d.recent.slice().reverse().map(tx => `
      <tr>
        <td>${tx.transaction_id}</td>
        <td>${(tx.fraud_probability * 100).toFixed(1)}%</td>
        <td><span class="badge badge-${tx.is_fraud ? 'fraud':'legit'}">${tx.is_fraud ? 'FRAUDE':'LÉGIT'}</span></td>
        <td><span class="badge badge-${tx.risk_level.toLowerCase()}">${tx.risk_level}</span></td>
        <td>${tx.latency_ms} ms</td>
      </tr>`).join('');
  } catch(e) {}
}

setInterval(refresh, 2000);
refresh();
</script>
</body>
</html>""")


@app.post("/predict", response_model=PredictionResponse)
async def predict(tx: TransactionRequest):
    t0 = time.time()

    # 1. Cache Redis
    if r is not None:
        try:
            cached = r.get(f"pred:{tx.transaction_id}")
            if cached:
                return PredictionResponse(**eval(cached))
        except Exception:
            pass

    # 2. Vecteur de features
    feat_map = {
        "V14":                tx.V14,
        "V17":                tx.V17,
        "velocity_1h_calc":   getattr(tx, "velocity_1h_calc", None) or tx.velocity_1h,
        "night_tx_ratio":     tx.night_tx_ratio,
        "zscore_amount_calc": getattr(tx, "zscore_amount_calc", None) or tx.zscore_amount or 0.0,
        "risk_score":         tx.risk_score or 0.1,
        "amount":             tx.amount,
        "V4":                 tx.V4 or 0.0,
        "V11":                tx.V11 or 0.0,
        "V12":                tx.V12 or 0.0,
    }
    X = np.array([[feat_map.get(f, 0.0) for f in FEATURES]], dtype=np.float32)

    # 3. Inférence
    proba    = float(MODEL.predict_proba(X)[0][1])
    is_fraud = proba >= THRESHOLD
    level    = "HIGH"   if proba > 0.70 \
          else "MEDIUM" if proba > 0.30 \
          else "LOW"
    latency  = round((time.time() - t0) * 1000, 1)

    result = {
        "transaction_id":    tx.transaction_id,
        "is_fraud":          is_fraud,
        "fraud_probability": round(proba, 4),
        "risk_level":        level,
        "model_version":     "p1-fraud-model/1",
        "latency_ms":        latency,
    }

    # 4. Mise à jour stats
    _stats["total"] += 1
    _stats["latency_sum"] += latency
    if is_fraud:
        _stats["fraud"] += 1
    else:
        _stats["legit"] += 1
    _recent.append(result)

    # 5. Cache Redis 1h
    if r is not None:
        try:
            r.setex(f"pred:{tx.transaction_id}", 3600, str(result))
        except Exception:
            pass

    return PredictionResponse(**result)
