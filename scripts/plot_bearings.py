"""
Plot the four bearings RMS over time to SEE the run-to-failure story.
Saves a PNG you can open. Investigation tool -> scripts/.

Run:  python scripts/plot_bearings.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("data/sensor_data_real.csv", parse_dates=["timestamp"])

plt.figure(figsize=(12, 6))
for bearing in sorted(df["device_id"].unique()):
    sub = df[df["device_id"] == bearing]
    plt.plot(sub["timestamp"], sub["value"], label=bearing, linewidth=1)

plt.title("NASA IMS Bearing - RMS vibration over 34-day run-to-failure (1st_test)")
plt.xlabel("Time")
plt.ylabel("RMS vibration (energy)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

out = Path("data") / "bearing_degradation.png"
plt.savefig(out, dpi=100)
print(f"Saved plot to {out}")
print("Open it from the data/ folder in VS Code (click the file).")
