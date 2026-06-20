"""Tests for drift detection."""

import numpy as np

from src.monitoring.drift import compute_drift, is_drifted


def test_identical_distributions_have_low_drift():
    rng = np.random.default_rng(0)
    ref = rng.normal(10, 1, size=500)
    cur = rng.normal(10, 1, size=500)  # same distribution
    assert compute_drift(ref, cur) < 0.2


def test_shifted_distribution_has_high_drift():
    rng = np.random.default_rng(0)
    ref = rng.normal(10, 1, size=500)
    cur = rng.normal(25, 1, size=500)  # shifted far -> big drift
    assert compute_drift(ref, cur) > 0.5


def test_drift_score_in_unit_range():
    rng = np.random.default_rng(1)
    ref = rng.normal(0, 1, size=200)
    cur = rng.normal(5, 1, size=200)
    score = compute_drift(ref, cur)
    assert 0.0 <= score <= 1.0


def test_is_drifted_respects_threshold():
    assert is_drifted(0.5, threshold=0.2) is True
    assert is_drifted(0.1, threshold=0.2) is False


def test_is_drifted_strictly_above_threshold():
    # Exactly at threshold is NOT drifted (strictly greater triggers).
    assert is_drifted(0.2, threshold=0.2) is False
