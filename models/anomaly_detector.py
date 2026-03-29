"""Train Isolation Forest + XGBoost scorer; export flagged transactions."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "artifacts"


def load_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], dict]:
    dt = pd.to_datetime(df["timestamp"])
    out = pd.DataFrame(
        {
            "amount": df["amount"].astype(float),
            "amount_log": np.log1p(df["amount"].astype(float)),
            "hour": dt.dt.hour.astype(int),
            "dow": dt.dt.dayofweek.astype(int),
        }
    )
    le_rail = LabelEncoder()
    out["payment_rail_enc"] = le_rail.fit_transform(df["payment_rail"].astype(str))
    le_send = LabelEncoder()
    out["sender_enc"] = le_send.fit_transform(df["sender_bic"].astype(str))
    encoders = {"payment_rail": le_rail, "sender_bic": le_send}
    feature_cols = list(out.columns)
    return out, feature_cols, encoders


def main() -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    csv_path = ARTIFACTS / "transactions.csv"
    if not csv_path.exists():
        raise SystemExit(f"Missing {csv_path}. Run: python data/generate_transactions.py")

    print("Loading transaction data...")
    df = pd.read_csv(csv_path)
    X, feature_cols, encoders = load_features(df)

    print("Training Isolation Forest...")
    contamination = 27 / len(df)
    iforest = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=42,
        n_jobs=-1,
    )
    iforest.fit(X)
    if_pred = iforest.predict(X)
    if_scores = -iforest.decision_function(X)

    print("Training XGBoost scorer...")
    y = df["is_anomaly"].astype(int).values
    xgb = XGBClassifier(
        n_estimators=120,
        max_depth=4,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        eval_metric="logloss",
    )
    xgb.fit(X, y)
    xgb_proba = xgb.predict_proba(X)[:, 1]

    # Normalize IF scores to [0,1] for blending
    s_min, s_max = if_scores.min(), if_scores.max()
    if_norm = (if_scores - s_min) / (s_max - s_min + 1e-9)
    combined = 0.55 * if_norm + 0.45 * xgb_proba
    top_idx = np.argsort(-combined)[:27]

    flagged = df.iloc[top_idx].copy()
    flagged["iforest_score"] = if_scores[top_idx]
    flagged["xgb_fraud_proba"] = xgb_proba[top_idx]
    flagged["combined_risk"] = combined[top_idx]

    joblib.dump(iforest, ARTIFACTS / "iforest.pkl")
    joblib.dump(xgb, ARTIFACTS / "xgb_scorer.pkl")
    joblib.dump(
        {"feature_cols": feature_cols, "encoders": encoders},
        ARTIFACTS / "detector_preprocess.pkl",
    )

    flagged_path = ARTIFACTS / "flagged_transactions.csv"
    flagged.to_csv(flagged_path, index=False)

    print("[OK] Models saved to artifacts/")
    print(f"[OK] Flagged {len(flagged)} transactions -> {flagged_path}")


if __name__ == "__main__":
    main()
