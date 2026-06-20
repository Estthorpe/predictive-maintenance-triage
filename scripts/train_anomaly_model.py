"""
Train the Isolation Forest anomaly detector on P7 sensor data and evaluate it
through the Phase-2 gate.

Pipeline:
  load data -> compute shared features -> time-based split ->
  SCALE features (fit on train only) -> train on healthy past ->
  score the later period -> run through the gate.

Feature scaling note: Isolation Forest's random splits are sensitive to each
feature's numeric range, so a large-range feature (e.g. fft_energy) can dominate
purely because of its scale, drowning the smaller-range pattern features that
actually carry the anomaly signal. StandardScaler puts every feature on equal
footing (mean 0, std 1). We FIT the scaler on training data only and apply it to
both train and test, so no test information leaks into training.

Run:  python scripts/train_anomaly_model.py
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.evaluation.metrics import evaluate_model_gate
from src.features.engineering import compute_features
from src.models.anomaly import AnomalyDetector

TRAIN_FRACTION = 0.60


def main() -> None:
    # 1. Load real data + labels
    readings = pd.read_csv("data/sensor_data.csv")
    labels_df = pd.read_csv("data/anomaly_labels.csv")
    values = readings["value"]
    labels = labels_df["is_anomaly"].astype(int).to_numpy()

    print(f"Loaded {len(values)} readings, {int(labels.sum())} true anomalies")

    # 2. Compute the SHARED features (same module serving will use)
    features = compute_features(values)
    print(f"Computed {features.shape[1]} features per reading")

    # 3. Time-based split (NO random shuffle - chronological)
    split = int(len(features) * TRAIN_FRACTION)
    train_feats = features.iloc[:split]
    test_feats = features.iloc[split:]
    test_labels = labels[split:]
    test_raw = values.to_numpy()[split:]
    print(
        f"Train rows: {len(train_feats)} (healthy past) | Test rows: {len(test_feats)} (later period)"
    )

    # 3b. SCALE the features. Fit on TRAIN ONLY, apply to both (no leakage).
    #     StandardScaler returns a NumPy array, so we re-wrap it as a DataFrame
    #     with the original column names (the detector expects a DataFrame).
    scaler = StandardScaler()
    train_scaled = pd.DataFrame(
        scaler.fit_transform(train_feats),  # learn mean/std from train + apply
        columns=train_feats.columns,
    )
    test_scaled = pd.DataFrame(
        scaler.transform(test_feats),  # apply the SAME scaling (no re-fit)
        columns=test_feats.columns,
    )

    # 4. Train on the healthy past (unsupervised - labels NOT used here)
    detector = AnomalyDetector(contamination=0.1, random_state=42).fit(train_scaled)

    # 5. Score the later period (higher = more anomalous)
    scores = detector.score(test_scaled)

    # 6. Flag the top ~10% most anomalous -> respects the FP budget by design.
    threshold = float(np.percentile(scores, 90))

    # 7. Run through the Phase-2 gate (the honest verdict)
    result = evaluate_model_gate(
        risk_scores=scores,
        true_labels=test_labels,
        raw_values=test_raw,
        threshold=threshold,
        recall_target=0.80,
        k_fraction=0.10,
        fp_budget=0.10,
    )

    print("\n=== EVALUATION GATE RESULT ===")
    print(f"recall@K:        {result.recall:.2f}")
    print(f"flag rate:       {result.flag_rate_value:.2f}")
    print(f"baseline recall: {result.baseline_recall:.2f}")
    print(f"PASSED:          {result.passed}")
    if not result.passed:
        print("Reasons it failed:")
        for r in result.reasons:
            print(f"  - {r}")


if __name__ == "__main__":
    main()
