"""
Anomaly-detection models for P7.

Isolation Forest: an unsupervised detector. It isolates points with random
splits; points that are EASY to isolate (few splits) are anomalies, points
that are HARD to isolate (many splits) are normal. Trained on FEATURES only -
no labels - because P7 has almost no labelled failures.

Score convention: this wrapper returns scores where HIGHER = MORE anomalous,
to match the evaluation gate. (scikit-learn's raw score is the opposite sign,
so we flip it - see below.)
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


class AnomalyDetector:
    """Thin, gate-friendly wrapper around scikit-learn's Isolation Forest."""

    def __init__(self, contamination: float = 0.1, random_state: int = 42):
        # contamination = expected fraction of anomalies; aligns with the
        # 10% FP budget. random_state fixes the seed for reproducibility.
        self.model = IsolationForest(
            contamination=contamination,
            random_state=random_state,
        )
        self._fitted = False

    def fit(self, features: pd.DataFrame) -> "AnomalyDetector":
        """Learn 'normal' from the feature table (no labels used)."""
        self.model.fit(features.to_numpy())
        self._fitted = True
        return self

    def score(self, features: pd.DataFrame) -> np.ndarray:
        """Return anomaly scores where HIGHER = MORE anomalous.

        sklearn's score_samples gives HIGHER = more normal, so we negate it
        to match the gate's 'higher = more suspicious' convention.
        """
        if not self._fitted:
            raise RuntimeError("Model must be fitted before scoring.")
        raw = self.model.score_samples(features.to_numpy())
        return -raw  # flip sign: now higher = more anomalous
