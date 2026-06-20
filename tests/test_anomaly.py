"""Tests for the AnomalyDetector (Isolation Forest wrapper)."""

import numpy as np
import pandas as pd
import pytest

from src.models.anomaly import AnomalyDetector


def _make_features(n=100, seed=0):
    """Mostly-normal feature data with a few obvious outliers at the end."""
    rng = np.random.default_rng(seed)
    normal = rng.normal(0, 1, size=(n, 3))
    feats = pd.DataFrame(normal, columns=["a", "b", "c"])
    # Plant 3 extreme outliers
    feats.iloc[-3:] = 50.0
    return feats


def test_score_requires_fit_first():
    det = AnomalyDetector()
    with pytest.raises(RuntimeError):
        det.score(_make_features())


def test_fit_returns_self():
    det = AnomalyDetector()
    assert det.fit(_make_features()) is det


def test_score_returns_one_value_per_row():
    feats = _make_features(n=50)
    det = AnomalyDetector().fit(feats)
    scores = det.score(feats)
    assert len(scores) == 50


def test_outliers_score_higher_than_normal():
    """The planted outliers should get higher anomaly scores than normal rows."""
    feats = _make_features(n=100)
    det = AnomalyDetector().fit(feats)
    scores = det.score(feats)
    # last 3 rows are the planted outliers
    outlier_mean = scores[-3:].mean()
    normal_mean = scores[:-3].mean()
    assert outlier_mean > normal_mean
