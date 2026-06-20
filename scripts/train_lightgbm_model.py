"""
Train the LightGBM supervised anomaly classifier and evaluate through the gate.

Pipeline:
  load data -> compute shared features -> time-based split ->
  train LightGBM on EARLY labels -> score the unseen LATER period ->
  run through the Phase-2 gate.

Leakage guard: the model trains ONLY on the early period; it is tested on the
strictly later period it never saw, so it must generalise, not memorise.

Run:  python scripts/train_lightgbm_model.py
"""

import numpy as np
import pandas as pd

from src.evaluation.metrics import evaluate_model_gate
from src.features.engineering import compute_features
from src.models.supervised import SupervisedAnomalyDetector

TRAIN_FRACTION = 0.80


def main() -> None:
    readings = pd.read_csv("data/sensor_data.csv")
    labels = pd.read_csv("data/anomaly_labels.csv")["is_anomaly"].astype(int).to_numpy()
    values = readings["value"]

    print(f"Loaded {len(values)} readings, {int(labels.sum())} true anomalies")

    features = compute_features(values)
    print(f"Computed {features.shape[1]} features per reading")

    # Time-based split (chronological - no shuffle)
    split = int(len(features) * TRAIN_FRACTION)
    train_feats, test_feats = features.iloc[:split], features.iloc[split:]
    train_labels, test_labels = labels[:split], labels[split:]
    test_raw = values.to_numpy()[split:]
    print(
        f"Train: {len(train_feats)} rows ({int(train_labels.sum())} anomalies) | "
        f"Test: {len(test_feats)} rows ({int(test_labels.sum())} anomalies)"
    )

    # Train LightGBM on EARLY labels (supervised); test on the UNSEEN later period.
    detector = SupervisedAnomalyDetector(random_state=42).fit(train_feats, train_labels)
    scores = detector.score(test_feats)

    threshold = float(np.percentile(scores, 90))  # flag top ~10%

    result = evaluate_model_gate(
        risk_scores=scores,
        true_labels=test_labels,
        raw_values=test_raw,
        threshold=threshold,
        recall_target=0.80,
        k_fraction=0.10,
        fp_budget=0.10,
    )

    print("\n=== EVALUATION GATE RESULT (LightGBM) ===")
    print(f"recall@K:        {result.recall:.2f}")
    print(f"flag rate:       {result.flag_rate_value:.2f}")
    print(f"baseline recall: {result.baseline_recall:.2f}")
    print(f"PASSED:          {result.passed}")
    if not result.passed:
        print("Reasons it failed:")
        for r in result.reasons:
            print(f"  - {r}")
    else:
        print("\nThe supervised model passed the gate on the hard data. ")


if __name__ == "__main__":
    main()
