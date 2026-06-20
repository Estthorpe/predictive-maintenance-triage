"""
Generate ground-truth anomaly labels for the real NASA bearing data using a
3-sigma control-chart rule (per bearing).

Logic (reasoned out before coding):
  1. Take each bearing's EARLY period (first 20%) as "healthy".
  2. Compute that healthy period's mean and std of RMS.
  3. Threshold = healthy_mean + 3 * healthy_std   (the control-chart alarm line).
  4. Label every reading: RMS > threshold -> anomaly (1), else normal (0).

These are HEURISTIC / weak labels (not human-inspected ground truth), but they
are defensible, reproducible, and standard industrial practice.

Output: data/anomaly_labels_real.csv  (timestamp, device_id, is_anomaly)

Run:  python scripts/generate_real_labels.py
"""

from pathlib import Path

import pandas as pd

HEALTHY_FRACTION = 0.20  # first 20% of each bearing's run = "healthy" baseline
SIGMA = 3.0  # 3-sigma control-chart rule


def main() -> None:
    df = pd.read_csv("data/sensor_data_real.csv", parse_dates=["timestamp"])
    df = df.sort_values(["device_id", "timestamp"]).reset_index(drop=True)

    labelled = []
    print(f"{'Bearing':<12}{'baseline_mean':<15}{'threshold':<12}{'anomalies':<10}")
    print("-" * 49)

    for bearing, sub in df.groupby("device_id"):
        sub = sub.copy()
        n_healthy = int(len(sub) * HEALTHY_FRACTION)
        healthy = sub["value"].iloc[:n_healthy]

        mean = healthy.mean()
        std = healthy.std()
        threshold = mean + SIGMA * std

        # Step 4: label every reading against the threshold
        sub["is_anomaly"] = (sub["value"] > threshold).astype(int)
        labelled.append(sub[["timestamp", "device_id", "is_anomaly"]])

        n_anom = int(sub["is_anomaly"].sum())
        print(f"{bearing:<12}{mean:<15.4f}{threshold:<12.4f}{n_anom:<10}")

    out_df = pd.concat(labelled).sort_values(["device_id", "timestamp"]).reset_index(drop=True)
    out = Path("data") / "anomaly_labels_real.csv"
    out_df.to_csv(out, index=False)

    total = int(out_df["is_anomaly"].sum())
    rate = total / len(out_df)
    print(f"\nWrote {len(out_df)} labels to {out}")
    print(f"Total anomalies: {total} ({rate:.1%} of all readings)")


if __name__ == "__main__":
    main()
