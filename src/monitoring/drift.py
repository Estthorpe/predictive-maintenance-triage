"""Data drift detection for P7 monitoring.

A deployed model fails SILENTLY when live data drifts away from the training
distribution. We detect this by comparing a reference sample (training data)
against a current sample (recent live data) using the Kolmogorov-Smirnov test,
and alarm when the drift score exceeds a threshold.
"""

import numpy as np
from scipy.stats import ks_2samp

DEFAULT_DRIFT_THRESHOLD = 0.2


def compute_drift(reference: np.ndarray, current: np.ndarray) -> float:
    """Drift score between a reference sample and a current sample.

    Uses the Kolmogorov-Smirnov statistic: 0.0 = identical distributions,
    1.0 = completely different. Higher = more drift.
    """
    result = ks_2samp(reference, current)
    return float(result.statistic)


def is_drifted(drift_score: float, threshold: float = DEFAULT_DRIFT_THRESHOLD) -> bool:
    """Has drift exceeded the threshold? (the number that triggers the alarm)."""
    return drift_score > threshold
