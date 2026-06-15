"""Tests for the shared time-domain feature module."""

import numpy as np
import pandas as pd

from src.features.engineering import (
    compute_features,
    compute_fft_features,
    compute_time_domain_features,
)


def test_output_has_one_row_per_reading():
    values = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    feats = compute_time_domain_features(values, window=3)
    assert len(feats) == len(values)


def test_expected_feature_columns_exist():
    values = pd.Series([1.0, 2.0, 3.0])
    feats = compute_time_domain_features(values, window=3)
    for col in ["roll_mean", "roll_std", "roll_min", "roll_max", "rate_of_change"]:
        assert col in feats.columns


def test_rolling_mean_uses_trailing_window():
    # window=2: mean at index 2 = mean of readings [index1, index2] = (2+3)/2 = 2.5
    values = pd.Series([1.0, 2.0, 3.0])
    feats = compute_time_domain_features(values, window=2)
    assert feats["roll_mean"].iloc[2] == 2.5


def test_rate_of_change_is_difference_from_previous():
    values = pd.Series([10.0, 12.0, 15.0])
    feats = compute_time_domain_features(values, window=2)
    assert feats["rate_of_change"].iloc[1] == 2.0  # 12 - 10
    assert feats["rate_of_change"].iloc[2] == 3.0  # 15 - 12


def test_no_nan_values_in_output():
    # Even the first rows (little history) must not contain NaN.
    values = pd.Series([5.0, 7.0, 9.0, 11.0])
    feats = compute_time_domain_features(values, window=3)
    assert not feats.isnull().any().any()


def test_no_future_leakage():
    """A feature at index i must not change if FUTURE values change."""
    base = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    changed_future = pd.Series([1.0, 2.0, 3.0, 99.0, 99.0])  # only indices 3,4 differ
    f_base = compute_time_domain_features(base, window=3)
    f_changed = compute_time_domain_features(changed_future, window=3)
    # Feature row at index 2 must be identical: it only sees the past (indices 0-2).
    assert f_base["roll_mean"].iloc[2] == f_changed["roll_mean"].iloc[2]


def test_fft_features_one_row_per_reading():
    values = pd.Series(np.sin(np.linspace(0, 10, 50)))
    feats = compute_fft_features(values, window=10)
    assert len(feats) == 50


def test_fft_feature_columns_exist():
    values = pd.Series(np.sin(np.linspace(0, 10, 50)))
    feats = compute_fft_features(values, window=10)
    for col in ["fft_energy", "fft_dominant_freq", "fft_entropy"]:
        assert col in feats.columns


def test_combined_features_has_all_columns():
    values = pd.Series(np.sin(np.linspace(0, 10, 50)))
    feats = compute_features(values, window=10)
    expected = [
        "roll_mean",
        "roll_std",
        "roll_min",
        "roll_max",
        "rate_of_change",
        "fft_energy",
        "fft_dominant_freq",
        "fft_entropy",
    ]
    for col in expected:
        assert col in feats.columns


def test_combined_features_no_nan():
    values = pd.Series(np.sin(np.linspace(0, 10, 50)))
    feats = compute_features(values, window=10)
    assert not feats.isnull().any().any()


def test_fft_no_future_leakage():
    base = pd.Series([1.0, 2.0, 1.5, 2.5, 1.0, 3.0, 1.2, 2.8])
    changed_future = base.copy()
    changed_future.iloc[6:] = 99.0  # change only the last readings
    f_base = compute_fft_features(base, window=3)
    f_changed = compute_fft_features(changed_future, window=3)
    # Feature at index 3 only sees indices 1-3 -> must be unchanged.
    assert f_base["fft_energy"].iloc[3] == f_changed["fft_energy"].iloc[3]


def test_louder_signal_has_more_fft_energy():
    quiet = pd.Series(np.sin(np.linspace(0, 10, 50)))
    loud = 10.0 * quiet
    f_quiet = compute_fft_features(quiet, window=10)
    f_loud = compute_fft_features(loud, window=10)
    # Energy is proportional to amplitude squared -> loud signal has more energy.
    assert f_loud["fft_energy"].mean() > f_quiet["fft_energy"].mean()
