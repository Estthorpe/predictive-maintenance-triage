"""
Diagnostic: why is recall stuck at 0.35? Look at WHERE the model's scores fall
for true anomalies vs normal readings. This is investigation, not production.

Run:  python scripts/diagnose_model.py
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.features.engineering import compute_features
from src.models.anomaly import AnomalyDetector

TRAIN_FRACTION = 0.60

readings = pd.read_csv("data/sensor_data.csv")
labels = pd.read_csv("data/anomaly_labels.csv")["is_anomaly"].astype(int).to_numpy()
values = readings["value"]

features = compute_features(values)
split = int(len(features) * TRAIN_FRACTION)

scaler = StandardScaler()
train_scaled = pd.DataFrame(scaler.fit_transform(features.iloc[:split]), columns=features.columns)
test_scaled = pd.DataFrame(scaler.transform(features.iloc[split:]), columns=features.columns)

detector = AnomalyDetector(contamination=0.1, random_state=42).fit(train_scaled)
scores = detector.score(test_scaled)
test_labels = labels[split:]

# Separate scores into "true anomaly" vs "normal"
anom_scores = scores[test_labels == 1]
norm_scores = scores[test_labels == 0]

print(f"Test set: {len(scores)} readings, {int(test_labels.sum())} true anomalies\n")
print("--- Score distribution (higher = more anomalous) ---")
print(f"Normal readings   -> mean {norm_scores.mean():.3f}, max {norm_scores.max():.3f}")
print(
    f"Anomaly readings  -> mean {anom_scores.mean():.3f}, min {anom_scores.min():.3f}, max {anom_scores.max():.3f}"
)

# The top-10% cutoff used by recall@K
k = max(1, int(round(0.10 * len(scores))))
cutoff = np.sort(scores)[::-1][k - 1]
print(f"\nTop-10% cutoff score: {cutoff:.3f}  (need to score ABOVE this to be 'caught')")

caught = int((anom_scores >= cutoff).sum())
print(f"True anomalies scoring above cutoff: {caught} / {int(test_labels.sum())}")

# Key question: of the MISSED anomalies, how close were they to the cutoff?
missed = anom_scores[anom_scores < cutoff]
if len(missed) > 0:
    print(f"\nMissed anomalies: {len(missed)}")
    print(f"  Their scores -> mean {missed.mean():.3f}, max {missed.max():.3f}")
    print(f"  How far below cutoff (max missed vs cutoff): {cutoff - missed.max():.3f}")
    print("\nINTERPRETATION:")
    print(
        "  - If missed scores are JUST below cutoff -> model ranks them OK, K is too small / events clustered."
    )
    print(
        "  - If missed scores are LOW (near normal mean) -> model genuinely cannot see these anomalies."
    )
