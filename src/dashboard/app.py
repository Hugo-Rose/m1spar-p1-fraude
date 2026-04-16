"""
Streamlit Dashboard — M1SPAR P1 Fraude J4
KPIs live, graphique fraudes rolling, formulaire test manuel.
Lance avec : streamlit run src/dashboard/app.py
"""
import time
import requests
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="M1SPAR — Fraud Detection",
    page_icon="🔍",
    layout="wide",
)

st.title("🔍 M1SPAR P1 — Fraud Detection Dashboard")
st.caption("Dashboard temps réel · Modèle XGBoost · MLflow Registry")

# ── Sidebar ───────────────────────────────────────────────────
st.sidebar.header("Configuration")
refresh_rate = st.sidebar.slider("Rafraîchissement (s)", 1, 10, 2)
st.sidebar.divider()
st.sidebar.header("Test Manuel")

with st.sidebar.form("predict_form"):
    tx_id       = st.text_input("Transaction ID", value=f"TX-{int(time.time())}")
    amount      = st.number_input("Montant (€)", value=150.0, min_value=0.01, max_value=49999.0)
    velocity    = st.slider("Velocity 1h", 0.0, 20.0, 2.0)
    night_ratio = st.slider("Night TX Ratio", 0.0, 1.0, 0.1)
    v14         = st.number_input("V14", value=0.02)
    v17         = st.number_input("V17", value=-0.05)
    submitted   = st.form_submit_button("Prédire", width="stretch")

if submitted:
    try:
        resp = requests.post(f"{API_URL}/predict", json={
            "transaction_id": tx_id,
            "amount": amount,
            "velocity_1h": velocity,
            "night_tx_ratio": night_ratio,
            "V14": v14,
            "V17": v17,
        }, timeout=5)
        result = resp.json()
        color = "red" if result["is_fraud"] else "green"
        label = "FRAUDE" if result["is_fraud"] else "LÉGITIME"
        st.sidebar.markdown(
            f"<div style='background:{'#450a0a' if result['is_fraud'] else '#052e16'};"
            f"padding:12px;border-radius:8px;margin-top:8px'>"
            f"<b style='color:{'#f87171' if result['is_fraud'] else '#4ade80'};font-size:1.1rem'>{label}</b><br>"
            f"Probabilité : {result['fraud_probability']*100:.1f}%<br>"
            f"Niveau : {result['risk_level']}<br>"
            f"Latence : {result['latency_ms']} ms"
            f"</div>",
            unsafe_allow_html=True
        )
    except Exception as e:
        st.sidebar.error(f"Erreur API : {e}")

# ── Fetch stats ───────────────────────────────────────────────
@st.cache_data(ttl=refresh_rate)
def get_stats():
    try:
        return requests.get(f"{API_URL}/stats", timeout=3).json()
    except Exception:
        return None

@st.cache_data(ttl=10)
def get_health():
    try:
        return requests.get(f"{API_URL}/health", timeout=3).json()
    except Exception:
        return None

stats  = get_stats()
health = get_health()

# ── Statut API ────────────────────────────────────────────────
if stats is None:
    st.error("API non disponible — lance : `uvicorn src.api.main:app --port 8000`")
    st.stop()

api_ok    = health is not None
redis_ok  = health.get("redis", False) if health else False
model_ver = health.get("model", "N/A") if health else "N/A"

col_s1, col_s2, col_s3 = st.columns(3)
col_s1.metric("API",   "✅ Online"  if api_ok   else "❌ Offline")
col_s2.metric("Redis", "✅ Online"  if redis_ok  else "⚠️ Offline")
col_s3.metric("Modèle", model_ver)

st.divider()

# ── KPIs ──────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total prédictions",  f"{stats['total_predictions']:,}")
c2.metric("Fraudes détectées",  f"{stats['fraud_count']:,}",
          delta=f"{stats['fraud_rate_pct']}%", delta_color="inverse")
c3.metric("Transactions légit", f"{stats['legit_count']:,}")
c4.metric("Latence moyenne",    f"{stats['avg_latency_ms']} ms")

st.divider()

# ── Graphiques ────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.subheader("Répartition Fraude / Légitime")
    if stats["total_predictions"] > 0:
        fig = go.Figure(go.Pie(
            labels=["Légitime", "Fraude"],
            values=[stats["legit_count"], stats["fraud_count"]],
            hole=0.6,
            marker_colors=["#4ade80", "#f87171"],
        ))
        fig.update_layout(
            paper_bgcolor="#0f172a", font_color="#e2e8f0",
            showlegend=True, height=280,
            margin=dict(t=10, b=10, l=10, r=10)
        )
        st.plotly_chart(fig, width="stretch")
    else:
        st.info("En attente de prédictions...")

with col2:
    st.subheader("Dernières transactions")
    recent = stats.get("recent", [])
    if recent:
        df = pd.DataFrame(recent[::-1])
        df["fraud_probability"] = (df["fraud_probability"] * 100).round(1).astype(str) + "%"
        df["is_fraud"] = df["is_fraud"].map({True: "🔴 FRAUDE", False: "🟢 LÉGIT"})
        df = df.rename(columns={
            "transaction_id": "ID",
            "is_fraud": "Résultat",
            "fraud_probability": "Probabilité",
            "risk_level": "Niveau",
            "latency_ms": "Latence (ms)",
        })[["ID", "Résultat", "Probabilité", "Niveau", "Latence (ms)"]]
        st.dataframe(df, width="stretch", height=260)
    else:
        st.info("En attente de prédictions...")

# ── Distribution des probabilités ────────────────────────────
recent = stats.get("recent", [])
if len(recent) >= 3:
    st.subheader("Distribution des probabilités de fraude")
    df_r = pd.DataFrame(recent)
    fig2 = px.histogram(
        df_r, x="fraud_probability", nbins=20,
        color="is_fraud",
        color_discrete_map={True: "#f87171", False: "#4ade80"},
        labels={"fraud_probability": "Probabilité", "is_fraud": "Fraude"},
    )
    fig2.update_layout(
        paper_bgcolor="#0f172a", plot_bgcolor="#1e293b",
        font_color="#e2e8f0", height=220,
        margin=dict(t=10, b=10, l=10, r=10)
    )
    st.plotly_chart(fig2, width="stretch")

# ── Auto-refresh ──────────────────────────────────────────────
st.caption(f"Dernière mise à jour : {datetime.now().strftime('%H:%M:%S')} · Rafraîchissement : {refresh_rate}s")
time.sleep(refresh_rate)
st.rerun()
