"""Generate synthetic payment transactions (normal + injected anomalies)."""

from __future__ import annotations

import random
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "artifacts"
RNG = random.Random(42)
np.random.seed(42)


def random_bic() -> str:
    bank = "".join(RNG.choices("ABCDEFGHIJKLMNOPQRSTUVWXYZ", k=4))
    country = RNG.choice(["US", "GB", "DE", "FR", "JP"])
    return f"{bank}{country}XXX"


def random_ts(start: datetime, end: datetime) -> datetime:
    delta = end - start
    return start + timedelta(seconds=RNG.randint(0, int(delta.total_seconds())))


def main() -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    out_path = ARTIFACTS / "transactions.csv"

    n_normal = 475
    n_anomaly = 25
    rows: list[dict] = []

    start = datetime(2024, 1, 1, tzinfo=None)
    end = datetime(2024, 6, 30, 23, 59, 59)

    print(f"Generating {n_normal} normal transactions...")
    tid_base = 1_000_000_000
    for i in range(n_normal):
        ts = random_ts(start, end)
        hour = ts.hour
        # Mostly business hours; occasional evening
        if RNG.random() < 0.12:
            hour = RNG.randint(18, 23)
            ts = ts.replace(hour=hour, minute=RNG.randint(0, 59))
        amount = round(RNG.uniform(25.0, 7500.0), 2)
        rows.append(
            {
                "transaction_id": f"TXN{tid_base + i:010d}",
                "amount": amount,
                "currency": "USD",
                "timestamp": ts.isoformat(),
                "sender_bic": random_bic(),
                "receiver_bic": random_bic(),
                "payment_rail": RNG.choice(["FEDWIRE", "ACH", "SWIFT", "RTP"]),
                "is_anomaly": 0,
            }
        )

    print(f"Injecting {n_anomaly} anomalous transactions...")
    patterns = [
        "near_threshold",
        "off_hours_wire",
        "structuring_pair",
        "velocity_burst",
        "sanctions_like",
        "round_trip",
        "high_value_night",
        "mule_pattern",
        "split_batch",
        "crypto_bridge",
    ]
    for j in range(n_anomaly):
        ts = random_ts(start, end)
        p = patterns[j % len(patterns)]
        if p == "near_threshold":
            amount = round(RNG.uniform(9995.0, 9999.99), 2)
            ts = ts.replace(hour=RNG.randint(9, 16))
        elif p == "off_hours_wire":
            amount = round(RNG.uniform(8000.0, 9500.0), 2)
            ts = ts.replace(hour=RNG.randint(1, 5), minute=RNG.randint(0, 59))
        elif p == "structuring_pair":
            amount = round(RNG.uniform(4990.0, 4999.0), 2)
        elif p == "velocity_burst":
            amount = round(RNG.uniform(3000.0, 6000.0), 2)
        elif p == "sanctions_like":
            amount = round(RNG.uniform(2000.0, 4000.0), 2)
        elif p == "round_trip":
            amount = round(RNG.uniform(15000.0, 25000.0), 2)
        elif p == "high_value_night":
            amount = round(RNG.uniform(12000.0, 18000.0), 2)
            ts = ts.replace(hour=RNG.randint(22, 23), minute=RNG.randint(0, 59))
        elif p == "mule_pattern":
            amount = round(RNG.uniform(500.0, 900.0), 2)
        elif p == "split_batch":
            amount = round(RNG.uniform(9900.0, 9999.0), 2)
        else:
            amount = round(RNG.uniform(7000.0, 11000.0), 2)

        rows.append(
            {
                "transaction_id": f"TXN{tid_base + n_normal + j:010d}",
                "amount": amount,
                "currency": "USD",
                "timestamp": ts.isoformat(),
                "sender_bic": random_bic(),
                "receiver_bic": random_bic(),
                "payment_rail": RNG.choice(["FEDWIRE", "SWIFT", "RTP"]),
                "is_anomaly": 1,
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"[OK] Saved {len(df)} transactions to {out_path}")


if __name__ == "__main__":
    main()
