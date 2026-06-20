"""
Diagnose WHY Isolation Forest only gets recall 0.48 on Bearing 4.
Look at where the model's scores fall for true anomalies vs normal, and whether
the misses are "scored low" (invisible) or "scored medium" (just below cutoff).

Run:  python scripts/diagnose_bearing4.py
"""

import numpy as np
import pandas as pd

from src.features.engineering import compute_features
from src.models.anomaly import AnomalyDetector

HEALTHY_FRACTION = 0.20

df = pd.read_csv("data/sensor_data_real.csv", parse_dates=["timestamp"])
labels_df = pd.read_csv("data/anomaly_labels_real.csv", parse_dates=["timestamp"])
merged = df.merge(labels_df, on=["timestamp", "device_id"])

sub = merged[merged["device_id"] == "BEARING_4"].sort_values("timestamp").reset_index(drop=True)
values = sub["value"]
labels = sub["is_anomaly"].to_numpy()

features = compute_features(values)
n_healthy = int(len(features) * HEALTHY_FRACTION)
detector = AnomalyDetector(contamination=0.05, random_state=42).fit(features.iloc[:n_healthy])
scores = detector.score(features)

anom_scores = scores[labels == 1]
norm_scores = scores[labels == 0]

k = max(1, int(0.10 * len(scores)))
cutoff = np.sort(scores)[::-1][k - 1]

print(f"Bearing 4: {len(scores)} readings, {int(labels.sum())} true anomalies")
print("\nScores (higher = more anomalous):")
print(f"  Normal   -> mean {norm_scores.mean():.3f}, max {norm_scores.max():.3f}")
print(
    f"  Anomaly  -> mean {anom_scores.mean():.3f}, min {anom_scores.min():.3f}, max {anom_scores.max():.3f}"
)
print(f"\nTop-10% cutoff: {cutoff:.3f}  (must score above this to be caught)")

caught = anom_scores[anom_scores >= cutoff]
missed = anom_scores[anom_scores < cutoff]
print(f"Caught: {len(caught)} / {int(labels.sum())}")
if len(missed) > 0:
    print(
        f"Missed: {len(missed)}  -> their scores: mean {missed.mean():.3f}, max {missed.max():.3f}"
    )
    gap = cutoff - missed.max()
    print(f"Closest miss is {gap:.3f} below the cutoff")
    print("\nINTERPRETATION:")
    print("  - Missed scores NEAR normal mean -> model can't see them (need better features).")
    print("  - Missed scores JUST below cutoff -> model ranks them OK (the anomalies are")
    print("    clustered, so 10% window can't hold them all - an evaluation/data shape issue).")

# Where in time are the anomalies? (clustered at the end, or spread out?)
anom_positions = np.where(labels == 1)[0]
print(f"\nAnomaly positions (row index out of {len(labels)}):")
print(
    f"  first at {anom_positions.min()}, last at {anom_positions.max()}, "
    f"spread across {anom_positions.max() - anom_positions.min()} rows"
)
