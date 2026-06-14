"""Tests for the evaluation metrics (recall@K, flag rate, FP budget, baseline)."""

import numpy as np

from src.evaluation.metrics import (
    flag_rate,
    naive_threshold_scores,
    recall_at_k,
    within_fp_budget,
)


def test_perfect_ranking_catches_all_failures():
    scores = np.array([0.9, 0.8, 0.1, 0.2, 0.3, 0.15, 0.05, 0.25, 0.35, 0.4])
    labels = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    assert recall_at_k(scores, labels, k_fraction=0.20) == 1.0


def test_worst_ranking_catches_nothing():
    scores = np.array([0.1, 0.2, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.25])
    labels = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    assert recall_at_k(scores, labels, k_fraction=0.20) == 0.0


def test_partial_catch():
    scores = np.array([0.9, 0.1, 0.8, 0.2, 0.3, 0.15, 0.05, 0.25, 0.35, 0.4])
    labels = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    assert recall_at_k(scores, labels, k_fraction=0.20) == 0.5


def test_no_failures_returns_zero():
    scores = np.array([0.9, 0.8, 0.1, 0.2])
    labels = np.array([0, 0, 0, 0])
    assert recall_at_k(scores, labels) == 0.0


def test_k_is_at_least_one():
    scores = np.array([0.9, 0.1, 0.2])
    labels = np.array([1, 0, 0])
    assert recall_at_k(scores, labels, k_fraction=0.10) == 1.0


def test_flag_rate_counts_assets_above_threshold():
    scores = np.array([0.9, 0.8, 0.1, 0.2, 0.3, 0.15, 0.05, 0.25, 0.35, 0.4])
    assert flag_rate(scores, threshold=0.5) == 0.2


def test_flag_rate_zero_when_nothing_above_threshold():
    scores = np.array([0.1, 0.2, 0.3, 0.4])
    assert flag_rate(scores, threshold=0.9) == 0.0


def test_within_budget_when_flag_rate_low():
    scores = np.array([0.9, 0.1, 0.2, 0.3, 0.4, 0.15, 0.05, 0.25, 0.35, 0.45])
    assert within_fp_budget(scores, threshold=0.5, budget=0.10) is True


def test_breach_when_flag_rate_too_high():
    scores = np.array([0.9, 0.8, 0.7, 0.6, 0.55, 0.1, 0.2, 0.3, 0.4, 0.05])
    assert within_fp_budget(scores, threshold=0.5, budget=0.10) is False


def test_exactly_at_budget_is_within():
    # 1 of 10 flagged = 10%, budget 10% -> within (<=)
    scores = np.array([0.9, 0.1, 0.2, 0.3, 0.4, 0.15, 0.05, 0.25, 0.35, 0.45])
    assert within_fp_budget(scores, threshold=0.5, budget=0.10) is True


def test_naive_baseline_normalises_to_unit_range():
    raw = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    scores = naive_threshold_scores(raw)
    assert scores.min() == 0.0
    assert scores.max() == 1.0


def test_naive_baseline_preserves_ranking():
    raw = np.array([5.0, 50.0, 15.0])
    scores = naive_threshold_scores(raw)
    assert np.argmax(scores) == 1


def test_naive_baseline_handles_identical_values():
    raw = np.array([10.0, 10.0, 10.0])
    scores = naive_threshold_scores(raw)
    assert np.all(scores == 0.0)
