"""
Test whether smoothing the anomaly scores improves Bearing 4 recall.

Reasoning: noisy NORMAL readings sometimes score high and steal top-10% slots.
A rolling average of the scores pulls lone noise-spikes DOWN (their neighbours
are normal) while keeping sustained real faults HIGH (their neighbours are also
high). So smoothing should separate noise from genuine degradation.

Run:  python scripts/smooth_bearing4.py
"""

import pandas as pd

from src.evaluation.metrics import recall_at_k
from src.features.engineering import compute_features
from src.models.anomaly import AnomalyDetector

HEALTHY_FRACTION = 0.20
SMOOTH_WINDOW = 10  # average each score with its ~10 recent neighbours


def recall_for(scores, labels):
    return recall_at_k(scores, labels, k_fraction=0.10)


df = pd.read_csv("data/sensor_data_real.csv", parse_dates=["timestamp"])
labels_df = pd.read_csv("data/anomaly_labels_real.csv", parse_dates=["timestamp"])
merged = df.merge(labels_df, on=["timestamp", "device_id"])

sub = merged[merged["device_id"] == "BEARING_4"].sort_values("timestamp").reset_index(drop=True)
values = sub["value"]
labels = sub["is_anomaly"].to_numpy()

features = compute_features(values)
n_healthy = int(len(features) * HEALTHY_FRACTION)
detector = AnomalyDetector(contamination=0.05, random_state=42).fit(features.iloc[:n_healthy])
raw_scores = detector.score(features)

# Smooth the scores with a trailing rolling average (no future leakage).
smoothed = pd.Series(raw_scores).rolling(window=SMOOTH_WINDOW, min_periods=1).mean().to_numpy()

print("=== Bearing 4: effect of smoothing the scores ===")
print(f"Raw scores      -> recall@K: {recall_for(raw_scores, labels):.2f}")
print(f"Smoothed scores -> recall@K: {recall_for(smoothed, labels):.2f}")
print(f"\n(smoothing window = {SMOOTH_WINDOW} readings)")
