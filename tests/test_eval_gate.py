"""
Tests for the evaluation GATE - the combined pass/fail rule that CI enforces.

Two kinds of test:
  1. Gate logic: feed known-good / known-bad scores and confirm pass/fail.
  2. Live wiring: run the gate against the real dataset (currently using the
     naive baseline as a stand-in model). In Phase 3 we point this at the real
     model. This keeps the CI evaluation gate live and demonstrable now.
"""

import numpy as np
import pandas as pd

from src.evaluation.metrics import evaluate_model_gate, naive_threshold_scores


def test_gate_passes_a_strong_model():
    # A genuinely strong model must satisfy ALL THREE conditions at once:
    #   - flag rate <= FP budget   (so inspection fraction == budget == 10%)
    #   - recall@K >= target
    #   - recall@K strictly BEATS the naive baseline
    #
    # 10 assets, 1 true failure. The MODEL ranks that failure #1 (score 0.95),
    # so at K=10% (top 1 asset) the model catches it -> recall 1.0.
    # The RAW values are arranged so the failure is NOT the largest raw value,
    # so the naive baseline misses it at K=10% -> baseline recall 0.0.
    # Threshold 0.9 flags only the single top score -> flag rate 0.10 (== budget).
    scores = np.array([0.95, 0.10, 0.20, 0.15, 0.05, 0.25, 0.30, 0.12, 0.08, 0.18])
    labels = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    raw = np.array([2.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 1.0, 2.5])
    result = evaluate_model_gate(
        scores,
        labels,
        raw,
        threshold=0.9,
        recall_target=0.80,
        k_fraction=0.10,
        fp_budget=0.10,
    )
    assert result.passed is True, result.reasons
    assert result.reasons == []


def test_gate_fails_a_weak_model():
    # Failures have the LOWEST scores -> recall@K will be 0 -> must fail the gate.
    scores = np.array([0.1, 0.2, 0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3])
    labels = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    raw = np.array([1, 2, 9, 8, 7, 6, 5, 4, 3, 2.5])
    result = evaluate_model_gate(
        scores, labels, raw, threshold=0.5, recall_target=0.80, k_fraction=0.20
    )
    assert result.passed is False
    assert len(result.reasons) > 0


def test_gate_runs_on_real_dataset():
    """Live wiring test: the gate executes end-to-end on the real data."""
    readings = pd.read_csv("data/sensor_data.csv")
    labels_df = pd.read_csv("data/anomaly_labels.csv")

    raw = readings["value"].to_numpy()
    labels = labels_df["is_anomaly"].astype(int).to_numpy()
    baseline_scores = naive_threshold_scores(raw)

    result = evaluate_model_gate(baseline_scores, labels, raw, threshold=0.5, recall_target=0.80)
    assert isinstance(result.passed, bool)
    assert 0.0 <= result.recall <= 1.0
    assert 0.0 <= result.flag_rate_value <= 1.0
