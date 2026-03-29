"""Streamlit dashboard: review flagged transactions and run the RAG explainer."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag.explainer_chain import explain_transaction

load_dotenv(ROOT / ".env")

st.set_page_config(page_title="Payment Anomaly Explainer", layout="wide")
st.title("Payment Anomaly Explainer")
st.caption("Isolation Forest + XGBoost flags · FAISS + Groq compliance explanations")

flagged_path = ROOT / "artifacts" / "flagged_transactions.csv"
tx_path = ROOT / "artifacts" / "transactions.csv"

if not flagged_path.exists() or not tx_path.exists():
    st.error(
        "Run the pipeline first: `python data/generate_transactions.py`, "
        "`python models/anomaly_detector.py`, `python models/embedder.py`."
    )
    st.stop()

flagged = pd.read_csv(flagged_path)
all_tx = pd.read_csv(tx_path)

st.subheader("Flagged transactions")
st.dataframe(flagged, use_container_width=True)

options = flagged["transaction_id"].astype(str).tolist()
choice = st.selectbox("Select a flagged transaction", options, index=0)
row = flagged[flagged["transaction_id"].astype(str) == choice].iloc[0]

detail = all_tx[all_tx["transaction_id"].astype(str) == choice]
extra = detail.iloc[0].to_dict() if len(detail) else {}

transaction_text = (
    f"Transaction {row.get('transaction_id', choice)}: "
    f"amount {row.get('amount')} {row.get('currency', 'USD')}, "
    f"rail {row.get('payment_rail')}, "
    f"time {row.get('timestamp')}, "
    f"sender {row.get('sender_bic')}, receiver {row.get('receiver_bic')}. "
    f"Model scores — IF: {float(row.get('iforest_score', 0)):.4f}, "
    f"XGB fraud proba: {float(row.get('xgb_fraud_proba', 0)):.4f}."
)

st.text_area("Context sent to explainer", transaction_text, height=120)

if st.button("Generate compliance explanation"):
    with st.spinner("Calling Groq + RAG..."):
        try:
            out = explain_transaction(transaction_text)
            st.markdown(out)
        except Exception as e:
            st.error(str(e))
