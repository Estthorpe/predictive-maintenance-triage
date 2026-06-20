"""
Generate BIAS-FREE labels using a time-based (RUL) rule, then test whether the
model (with the z-score feature) can beat the raw-RMS threshold on a FAIR test.

Why this is fair: the previous labels were made from raw RMS (3-sigma), so a
raw-RMS threshold trivially won. These NEW labels depend only on WHEN a reading
occurred (final fraction of the bearing's life = degradation), independent of
RMS magnitude. Now the threshold has no built-in advantage.

Honest scoping:
  - Only bearings that ACTUALLY failed get end-of-life labels (3 and 4).
  - Degradation window kept SHORT (final ~10%) so the anomaly rate stays sensible
    and recall@K@10% remains meaningful.

Run:  python scripts/rul_labels_compare.py
"""

import numpy as np
import pandas as pd

from src.evaluation.metrics import naive_threshold_scores, recall_at_k
from src.features.engineering import compute_features
from src.models.anomaly import AnomalyDetector

HEALTHY_FRACTION = 0.20  # for training the model + the z-score baseline
DEGRADATION_FRACTION = 0.10  # final 10% of life = "degradation" (the new labels)
FAILED_BEARINGS = ["BEARING_3", "BEARING_4"]  # only these actually failed


df = pd.read_csv("data/sensor_data_real.csv", parse_dates=["timestamp"])


def run_bearing(name: str) -> None:
    sub = df[df["device_id"] == name].sort_values("timestamp").reset_index(drop=True)
    values = sub["value"]
    n = len(values)

    # NEW time-based labels: final DEGRADATION_FRACTION of the run = anomaly (1).
    labels = np.zeros(n, dtype=int)
    start = int(n * (1 - DEGRADATION_FRACTION))
    labels[start:] = 1
    rate = labels.mean()

    # Features (+ per-bearing z-score deviation feature)
    features = compute_features(values)
    n_healthy = int(n * HEALTHY_FRACTION)
    healthy = values.iloc[:n_healthy]
    zscore = (values - healthy.mean()) / healthy.std()
    feats_z = features.copy()
    feats_z["baseline_zscore"] = zscore.to_numpy()

    def model_recall(feats):
        det = AnomalyDetector(contamination=0.05, random_state=42).fit(feats.iloc[:n_healthy])
        return recall_at_k(det.score(feats), labels, k_fraction=0.10)

    threshold_recall = recall_at_k(naive_threshold_scores(values.to_numpy()), labels, 0.10)

    print(f"\n=== {name}  (time-based labels, {int(labels.sum())} anomalies = {rate:.0%}) ===")
    print(f"Threshold (raw RMS) recall@K:        {threshold_recall:.2f}   <- bar to beat")
    print(f"Model WITHOUT z-score:               {model_recall(features):.2f}")
    print(f"Model WITH z-score feature:          {model_recall(feats_z):.2f}")


def main() -> None:
    print("Labels now depend on TIME (end-of-life), independent of RMS magnitude.")
    for b in FAILED_BEARINGS:
        run_bearing(b)


if __name__ == "__main__":
    main()
