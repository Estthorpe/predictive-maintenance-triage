"""
Supervised anomaly classifier for P7 (LightGBM).

Unlike the unsupervised Isolation Forest (which only sees features and assumes
anomalies are spatial outliers), LightGBM is SHOWN labelled examples and learns
the actual signature of an anomaly - e.g. the specific combination of
rate-of-change, rolling std, and FFT features that marks a drift/volatility
fault. This matches P7 anomalies, which are subtle feature-patterns rather than
obvious outliers.

Leakage guard: trained on the EARLY period only; tested on a strictly later,
unseen period (handled by the time-based split in the training script).

Score convention: returns the probability of the anomaly class (0..1), which is
already in the gate's "higher = more anomalous" language - no sign flip needed.
"""

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier


class SupervisedAnomalyDetector:
    """LightGBM classifier that learns the anomaly pattern from labelled features."""

    def __init__(self, random_state: int = 42):
        self.model = LGBMClassifier(random_state=random_state, verbose=-1)
        self._fitted = False

    def fit(self, features: pd.DataFrame, labels: np.ndarray) -> "SupervisedAnomalyDetector":
        """Learn the anomaly pattern from labelled training data."""
        self.model.fit(features, labels)
        self._fitted = True
        return self

    def score(self, features: pd.DataFrame) -> np.ndarray:
        """Return anomaly probability per row (higher = more anomalous)."""
        if not self._fitted:
            raise RuntimeError("Model must be fitted before scoring.")
        return self.model.predict_proba(features)[:, 1]  # probability of class 1 (anomaly)
