"""
Run the (existing) unsupervised Isolation Forest on REAL NASA bearing data.

Per bearing:
  - train on the HEALTHY early period (first 20%) - unsupervised, no labels
  - score the full run (higher = more anomalous)
  - evaluate against the control-chart labels via the Phase-2 gate

Real failing bearings ARE genuine outliers from their healthy baseline, so the
unsupervised model that struggled on synthetic data should now succeed here.

Run:  python scripts/train_real_anomaly.py
"""

import numpy as np
import pandas as pd

from src.evaluation.metrics import evaluate_model_gate
from src.features.engineering import compute_features
from src.models.anomaly import AnomalyDetector

HEALTHY_FRACTION = 0.20


def evaluate_bearing(values: pd.Series, labels: np.ndarray, name: str) -> None:
    features = compute_features(values)

    n_healthy = int(len(features) * HEALTHY_FRACTION)
    train_feats = features.iloc[:n_healthy]  # healthy early period

    detector = AnomalyDetector(contamination=0.05, random_state=42).fit(train_feats)
    scores = detector.score(features)  # score the WHOLE run

    raw = values.to_numpy()
    # Flag AT MOST the top 10% (n assets). Pick the threshold just below the
    # k-th highest score so flag_rate (strictly-greater) lands <= 0.10.
    k = max(1, int(0.10 * len(scores)))
    kth_highest = np.sort(scores)[::-1][k - 1]
    threshold = float(np.nextafter(kth_highest, -np.inf))  # just below -> exactly k flagged

    result = evaluate_model_gate(
        risk_scores=scores,
        true_labels=labels,
        raw_values=raw,
        threshold=threshold,
        recall_target=0.80,
        k_fraction=0.10,
        fp_budget=0.10,
    )

    n_anom = int(labels.sum())
    print(f"\n=== {name}  ({n_anom} labelled anomalies) ===")
    print(f"recall@K:        {result.recall:.2f}")
    print(f"flag rate:       {result.flag_rate_value:.2f}")
    print(f"baseline recall: {result.baseline_recall:.2f}")
    print(f"PASSED:          {result.passed}")
    if not result.passed:
        for r in result.reasons:
            print(f"  - {r}")


def main() -> None:
    df = pd.read_csv("data/sensor_data_real.csv", parse_dates=["timestamp"])
    labels_df = pd.read_csv("data/anomaly_labels_real.csv", parse_dates=["timestamp"])

    merged = df.merge(labels_df, on=["timestamp", "device_id"])

    for bearing in sorted(merged["device_id"].unique()):
        sub = merged[merged["device_id"] == bearing].sort_values("timestamp")
        if int(sub["is_anomaly"].sum()) == 0:
            print(f"\n=== {bearing} === skipped (0 anomalies - stayed healthy)")
            continue
        evaluate_bearing(
            sub["value"].reset_index(drop=True),
            sub["is_anomaly"].to_numpy(),
            bearing,
        )


if __name__ == "__main__":
    main()
