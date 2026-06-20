"""
Train the anomaly detector ONCE and save it as a serving artifact.

The artifact is a BUNDLE (not just the model): it carries everything the serving
API needs to reproduce training-time features exactly -
  - the fitted Isolation Forest model
  - the per-asset healthy baseline (mean, std) for the z-score feature
  - feature config (window size)
Bundling these together is what prevents training-serving skew: serving cannot
accidentally use a different baseline or window than training did.

Output: models/anomaly_model.joblib

Run:  python scripts/train_and_save_model.py
"""

from pathlib import Path

import joblib
import pandas as pd

from src.features.engineering import DEFAULT_WINDOW, compute_features
from src.models.anomaly import AnomalyDetector

HEALTHY_FRACTION = 0.20


def main() -> None:
    # Use the synthetic dataset as the training source (reproducible, in-repo).
    readings = pd.read_csv("data/sensor_data.csv")
    values = readings["value"]

    # 1. Healthy baseline (first 20%) - the stats the z-score feature needs.
    n_healthy = int(len(values) * HEALTHY_FRACTION)
    healthy = values.iloc[:n_healthy]
    baseline_mean = float(healthy.mean())
    baseline_std = float(healthy.std())

    # 2. Compute features and train on the healthy period (unsupervised).
    features = compute_features(values)
    detector = AnomalyDetector(contamination=0.1, random_state=42).fit(features.iloc[:n_healthy])

    # 3. Bundle EVERYTHING serving needs into one artifact.
    bundle = {
        "model": detector,
        "baseline_mean": baseline_mean,
        "baseline_std": baseline_std,
        "window": DEFAULT_WINDOW,
        "feature_names": list(features.columns),
    }

    out_dir = Path("models")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "anomaly_model.joblib"
    joblib.dump(bundle, out_path)

    print(f"Saved model artifact to {out_path}")
    print(f"  baseline_mean: {baseline_mean:.4f}")
    print(f"  baseline_std:  {baseline_std:.4f}")
    print(f"  window:        {DEFAULT_WINDOW}")
    print(f"  features:      {len(features.columns)}")
    print("\nThe serving API will LOAD this bundle at startup and reuse it per request.")


if __name__ == "__main__":
    main()
