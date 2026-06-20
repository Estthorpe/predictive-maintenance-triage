"""
Diagnostic: is LightGBM stuck at 0.48 because train and test contain different
anomaly TYPES (drift vs volatility), so it must recognise a kind it under-saw?

We rebuild the same events the generator made, label each as drift/volatility,
and check how they fall across the train/test split.

Run:  python scripts/diagnose_split.py
"""

import numpy as np

# Re-derive the event layout EXACTLY as the generator made it (same seed/params).
SEED = 42
N_READINGS = 2000
DEGRADATION_START = 1200
N_ANOMALY_EVENTS = 8
ANOMALY_LENGTH = 12
TRAIN_FRACTION = 0.80

rng = np.random.default_rng(SEED)
# Reproduce the SAME random draws the generator used, in the same order:
# (baseline noise draws happen first in the generator, so we must mirror them)
for i in range(N_READINGS):
    _ = rng.normal(0, 1.0)  # consume the per-reading noise draws (same as generator)

possible_starts = range(DEGRADATION_START, N_READINGS - ANOMALY_LENGTH)
event_starts = rng.choice(list(possible_starts), size=N_ANOMALY_EVENTS, replace=False)

split = int(N_READINGS * TRAIN_FRACTION)

print(f"Split point: row {split} (train < {split} <= test)\n")
print(f"{'Event':<6}{'StartRow':<10}{'Type':<12}{'Side':<6}")
print("-" * 36)
for k, s in enumerate(sorted(event_starts)):
    etype = "drift" if k % 2 == 0 else "volatility"
    side = "TRAIN" if s < split else "TEST"
    print(f"{k:<6}{s:<10}{etype:<12}{side:<6}")

# Summary counts
train_events = [(k, s) for k, s in enumerate(sorted(event_starts)) if s < split]
test_events = [(k, s) for k, s in enumerate(sorted(event_starts)) if s >= split]


def types(evs):
    return [("drift" if k % 2 == 0 else "volatility") for k, _ in evs]


print(f"\nTRAIN events: {len(train_events)} -> types: {types(train_events)}")
print(f"TEST  events: {len(test_events)} -> types: {types(test_events)}")
print("\nINTERPRETATION:")
print("  - If TEST has a type that TRAIN barely/never had -> the model cannot")
print("    recognise it -> data/sampling problem -> fix the split or add events.")
