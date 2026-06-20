"""
Load the real NASA IMS bearing data (1st_test) directly from data/archive.zip,
WITHOUT extracting ~10 GB to disk.

For each timestamped file (1 second of 20 kHz vibration, 8 channels = 4 bearings
x 2 accelerometers), compute RMS (vibration energy) per bearing. Rising RMS over
time = a bearing wearing out.

Output: data/sensor_data_real.csv  (timestamp, device_id, sensor_type, value)

Run:  python scripts/load_nasa_bearing.py
"""

import io
import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# Point DIRECTLY at the bearing zip - no auto-discovery (avoids grabbing the
# wrong file). This is the Kaggle download you placed in data/.
ZIP_PATH = Path("data") / "archive.zip"
if not ZIP_PATH.exists():
    raise FileNotFoundError(
        f"{ZIP_PATH} not found. Confirm the bearing zip is at data\\archive.zip"
    )

INNER_PREFIX = "1st_test/1st_test/"
N_BEARINGS = 4


def parse_timestamp(name: str) -> datetime:
    base = name.split("/")[-1]
    return datetime.strptime(base, "%Y.%m.%d.%H.%M.%S")


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x**2)))


def main() -> None:
    rows = []
    with zipfile.ZipFile(ZIP_PATH) as zf:
        entries = [e for e in zf.namelist() if e.startswith(INNER_PREFIX) and not e.endswith("/")]
        entries.sort()
        total = len(entries)
        print(f"Using zip: {ZIP_PATH}")
        print(f"Found {total} timestamped files. Processing...")

        if total == 0:
            raise RuntimeError(
                "0 files matched the '1st_test/1st_test/' path. "
                "The zip layout may differ - tell Claude what the entries look like."
            )

        for i, name in enumerate(entries):
            with zf.open(name) as f:
                data = np.loadtxt(io.TextIOWrapper(f, encoding="utf-8"))
            ts = parse_timestamp(name)
            for b in range(N_BEARINGS):
                ch = data[:, 2 * b : 2 * b + 2].mean(axis=1)
                rows.append(
                    {
                        "timestamp": ts,
                        "device_id": f"BEARING_{b + 1}",
                        "sensor_type": "vibration",
                        "value": round(rms(ch), 6),
                    }
                )
            if (i + 1) % 500 == 0:
                print(f"  processed {i + 1}/{total} files...")

    df = pd.DataFrame(rows).sort_values(["device_id", "timestamp"]).reset_index(drop=True)
    out = Path("data") / "sensor_data_real.csv"
    df.to_csv(out, index=False)

    print(f"\nWrote {len(df)} rows to {out}")
    print(
        f"Bearings: {df['device_id'].nunique()} | Span: {df['timestamp'].min()} -> {df['timestamp'].max()}"
    )
    print("\nRMS per bearing (low early = healthy, high late = degraded):")
    print(df.groupby("device_id")["value"].agg(["min", "max", "mean"]).round(4).to_string())


if __name__ == "__main__":
    main()
