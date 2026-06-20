"""Tests for the SupervisedAnomalyDetector (LightGBM)."""

import numpy as np
import pandas as pd
import pytest

from src.models.supervised import SupervisedAnomalyDetector


def _make_labelled_data(n=200, seed=0):
    """Features where the label is driven by a clear pattern in column 'a'."""
    rng = np.random.default_rng(seed)
    feats = pd.DataFrame(
        {
            "a": rng.normal(0, 1, n),
            "b": rng.normal(0, 1, n),
            "c": rng.normal(0, 1, n),
        }
    )
    # Anomaly when 'a' is high - a learnable pattern.
    labels = (feats["a"] > 1.0).astype(int).to_numpy()
    return feats, labels


def test_score_requires_fit_first():
    det = SupervisedAnomalyDetector()
    feats, _ = _make_labelled_data()
    with pytest.raises(RuntimeError):
        det.score(feats)


def test_fit_returns_self():
    det = SupervisedAnomalyDetector()
    feats, labels = _make_labelled_data()
    assert det.fit(feats, labels) is det


def test_score_returns_probability_per_row():
    feats, labels = _make_labelled_data(n=150)
    det = SupervisedAnomalyDetector().fit(feats, labels)
    scores = det.score(feats)
    assert len(scores) == 150
    assert scores.min() >= 0.0 and scores.max() <= 1.0  # valid probabilities


def test_learns_the_pattern():
    """Rows that match the anomaly pattern should score higher than those that don't."""
    feats, labels = _make_labelled_data(n=300)
    det = SupervisedAnomalyDetector().fit(feats, labels)
    scores = det.score(feats)
    anomaly_mean = scores[labels == 1].mean()
    normal_mean = scores[labels == 0].mean()
    assert anomaly_mean > normal_mean
