"""
Synthetic run-to-failure sensor data generator (P7 C3 fallback) - v2.

v2 makes anomalies REALISTIC rather than trivial. Real bearing faults are not
just "big spikes" - they show up as PATTERN changes:
  - drift anomalies      : an abnormal local upward trend (caught by rate-of-change
                           / rolling mean, NOT by raw magnitude)
  - volatility anomalies : the signal becomes erratic / jittery (caught by rolling
                           std and FFT, NOT by raw magnitude)

Because these anomalies are NOT simply the largest raw values, a naive
"flag the biggest value" baseline can no longer catch them - so a feature-based
model has to earn its keep. The anomaly positions are recorded as ground truth.

Outputs:
  data/sensor_data.csv     -> timestamp, device_id, sensor_type, value
  data/anomaly_labels.csv  -> timestamp, device_id, is_anomaly

Run:  python scripts/generate_sensor_data.py
"""

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
DEVICE_ID = "BEARING_07"
SENSOR_TYPE = "vibration"

N_READINGS = 2000
DEGRADATION_START = 1200
N_ANOMALY_EVENTS = 20  # number of anomalous EVENTS (each spans several readings)
ANOMALY_LENGTH = 4  # readings per anomalous event

HEALTHY_BASELINE = 10.0
HEALTHY_NOISE = 1.0
DEGRADATION_SLOPE = 0.015  # mild overall wear trend


def generate() -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(SEED)

    start_time = datetime(2026, 1, 1, 0, 0, 0)
    timestamps = [start_time + timedelta(minutes=i) for i in range(N_READINGS)]

    # Base signal: gentle wobble + mild wear drift after degradation starts.
    values = np.zeros(N_READINGS)
    for i in range(N_READINGS):
        baseline = HEALTHY_BASELINE
        if i >= DEGRADATION_START:
            baseline += DEGRADATION_SLOPE * (i - DEGRADATION_START)
        values[i] = baseline + rng.normal(0, HEALTHY_NOISE)

    is_anomaly = np.zeros(N_READINGS, dtype=bool)

    # Place anomalous EVENTS in the degradation phase. Each event is either a
    # local drift or a volatility burst - both are PATTERN anomalies, not spikes.
    possible_starts = range(DEGRADATION_START, N_READINGS - ANOMALY_LENGTH)
    event_starts = rng.choice(list(possible_starts), size=N_ANOMALY_EVENTS, replace=False)

    for k, s in enumerate(event_starts):
        e = s + ANOMALY_LENGTH
        if k % 2 == 0:
            # DRIFT anomaly: a gentle abnormal ramp (small per-step, NOT a big value)
            ramp = np.linspace(0, 4.0, ANOMALY_LENGTH)  # max +4 over 12 readings
            values[s:e] += ramp
        else:
            # VOLATILITY anomaly: extra jitter, but mean stays similar (pattern, not size)
            values[s:e] += rng.normal(0, 4.0, size=ANOMALY_LENGTH)
        is_anomaly[s:e] = True

    # Keep everything inside the contract range [0, 100].
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
        {"timestamp": timestamps, "device_id": DEVICE_ID, "is_anomaly": is_anomaly}
    )
    return readings, labels


def main() -> None:
    readings, labels = generate()
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    readings.to_csv(data_dir / "sensor_data.csv", index=False)
    labels.to_csv(data_dir / "anomaly_labels.csv", index=False)

    n_anom = int(labels["is_anomaly"].sum())
    print(f"Wrote {len(readings)} readings to data/sensor_data.csv")
    print(
        f"Wrote {n_anom} anomalous readings ({N_ANOMALY_EVENTS} events) to data/anomaly_labels.csv"
    )
    print("Anomaly types: gradual drift + volatility bursts (PATTERN anomalies, not spikes)")

    # Quick honesty check: are anomalies the biggest raw values? They should NOT be.
    vals = readings["value"].to_numpy()
    is_anom = labels["is_anomaly"].to_numpy()
    top10pct_threshold = np.percentile(vals, 90)
    anomalies_in_top10 = (vals[is_anom] >= top10pct_threshold).mean()
    print(f"\nFraction of anomalies that are in the top-10% raw values: {anomalies_in_top10:.0%}")
    print("(Low number = good: raw magnitude alone can no longer find them.)")


if __name__ == "__main__":
    main()
