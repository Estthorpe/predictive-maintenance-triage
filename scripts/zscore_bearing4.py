"""
Test whether a per-bearing baseline-deviation feature lets the model beat the
raw-RMS threshold on Bearing 4 (the subtle case).

Idea: instead of "is RMS high in absolute terms?", measure "how many standard
deviations is this reading above THIS bearing's own healthy baseline?" (a
z-score). This is a NOISE-AWARE, PER-ASSET deviation signal that can catch a
subtle relative rise the absolute threshold misses.

Run:  python scripts/zscore_bearing4.py
"""

import pandas as pd

from src.evaluation.metrics import naive_threshold_scores, recall_at_k
from src.features.engineering import compute_features
from src.models.anomaly import AnomalyDetector

HEALTHY_FRACTION = 0.20


df = pd.read_csv("data/sensor_data_real.csv", parse_dates=["timestamp"])
labels_df = pd.read_csv("data/anomaly_labels_real.csv", parse_dates=["timestamp"])
merged = df.merge(labels_df, on=["timestamp", "device_id"])

sub = merged[merged["device_id"] == "BEARING_4"].sort_values("timestamp").reset_index(drop=True)
values = sub["value"]
labels = sub["is_anomaly"].to_numpy()

# Existing engineered features
features = compute_features(values)
n_healthy = int(len(features) * HEALTHY_FRACTION)

# --- NEW FEATURE: per-bearing baseline deviation (z-score) ---
# Baseline = healthy early period of THIS bearing.
healthy_vals = values.iloc[:n_healthy]
base_mean = healthy_vals.mean()
base_std = healthy_vals.std()
# z = how many std devs above the bearing's own healthy mean (noise-aware).
zscore = (values - base_mean) / base_std
features_with_z = features.copy()
features_with_z["baseline_zscore"] = zscore.to_numpy()


# Helper to train + score + get recall, given a feature set
def recall_with(feats):
    detector = AnomalyDetector(contamination=0.05, random_state=42).fit(feats.iloc[:n_healthy])
    scores = detector.score(feats)
    return recall_at_k(scores, labels, k_fraction=0.10)


threshold_recall = recall_at_k(naive_threshold_scores(values.to_numpy()), labels, k_fraction=0.10)

print("=== Bearing 4: can the z-score feature beat the threshold? ===")
print(f"Threshold (raw RMS) recall@K:        {threshold_recall:.2f}   <- the bar to beat")
print(f"Model WITHOUT z-score feature:       {recall_with(features):.2f}")
print(f"Model WITH baseline z-score feature: {recall_with(features_with_z):.2f}")
