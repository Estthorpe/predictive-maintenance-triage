"""
Synthetic run-to-failure sensor data generator (P7 C3 fallback).

Produces a realistic bearing-degradation trajectory:
  - a HEALTHY phase: vibration wobbles gently around a low baseline
  - a DEGRADATION phase: the baseline drifts upward over time (wear)
  - injected ANOMALIES: sharp spikes during degradation (developing damage),
    whose positions we record as ground truth

Outputs:
  data/sensor_data.csv     -> timestamp, device_id, sensor_type, value
  data/anomaly_labels.csv  -> timestamp, device_id, is_anomaly  (the answer key)

Run:  python scripts/generate_sensor_data.py
"""

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# Fixed seed -> identical data every run (reproducibility).
SEED = 42
DEVICE_ID = "BEARING_07"
SENSOR_TYPE = "vibration"

N_READINGS = 2000  # total readings in the run
DEGRADATION_START = 1200  # index where wear begins (healthy before this)
N_ANOMALIES = 15  # number of injected spikes during degradation

HEALTHY_BASELINE = 10.0  # normal vibration level
HEALTHY_NOISE = 1.0  # gentle wobble around the baseline
DEGRADATION_SLOPE = 0.02  # how fast the baseline rises per reading after start
ANOMALY_SPIKE = 25.0  # how far above trend an anomaly jumps


def generate() -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(SEED)

    start_time = datetime(2026, 1, 1, 0, 0, 0)
    timestamps = [start_time + timedelta(minutes=i) for i in range(N_READINGS)]

    values = np.zeros(N_READINGS)
    for i in range(N_READINGS):
        baseline = HEALTHY_BASELINE
        # After degradation starts, the baseline drifts upward (wear accumulates).
        if i >= DEGRADATION_START:
            baseline += DEGRADATION_SLOPE * (i - DEGRADATION_START)
        # Gentle random wobble around the baseline (normal operation noise).
        values[i] = baseline + rng.normal(0, HEALTHY_NOISE)

    # Inject anomalies: sharp spikes, only during the degradation phase.
    anomaly_indices = rng.choice(
        range(DEGRADATION_START, N_READINGS), size=N_ANOMALIES, replace=False
    )
    is_anomaly = np.zeros(N_READINGS, dtype=bool)
    for idx in anomaly_indices:
        values[idx] += ANOMALY_SPIKE
        is_anomaly[idx] = True

    # Keep all values inside the contract range [0, 100] for vibration.
    values = np.clip(values, 0.0, 100.0)

    readings = pd.DataFrame(
        {
            "timestamp": timestamps,
            "device_id": DEVICE_ID,
            "sensor_type": SENSOR_TYPE,
            "value": np.round(values, 3),
        }
    )

    labels = pd.DataFrame(
        {
            "timestamp": timestamps,
            "device_id": DEVICE_ID,
            "is_anomaly": is_anomaly,
        }
    )

    return readings, labels


def main() -> None:
    readings, labels = generate()

    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    readings.to_csv(data_dir / "sensor_data.csv", index=False)
    labels.to_csv(data_dir / "anomaly_labels.csv", index=False)

    print(f"Wrote {len(readings)} readings to data/sensor_data.csv")
    print(f"Wrote {int(labels['is_anomaly'].sum())} anomalies to data/anomaly_labels.csv")
    print(f"Healthy phase: rows 0-{DEGRADATION_START - 1}")
    print(f"Degradation phase: rows {DEGRADATION_START}-{N_READINGS - 1}")
    print("\nFirst 3 rows:")
    print(readings.head(3).to_string(index=False))
    print("\nLast 3 rows (degraded, higher values):")
    print(readings.tail(3).to_string(index=False))


if __name__ == "__main__":
    main()
