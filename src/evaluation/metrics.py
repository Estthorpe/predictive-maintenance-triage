"""
Evaluation metrics and the model gate for predictive-maintenance anomaly detection.

These metrics are MODEL-AGNOSTIC: they judge a set of risk scores against the
true labels, regardless of which model produced the scores, so the same gate
judges every candidate model fairly.

  - recall_at_k        : the GOAL  - fraction of true failures caught in top-K
  - flag_rate          : helper    - fraction of assets flagged at a threshold
  - within_fp_budget   : the LIMIT - is the flag rate within budget?
  - naive_threshold_scores : the BENCHMARK any real model must beat
  - evaluate_model_gate    : combines all three into one pass/fail verdict
"""

from dataclasses import dataclass

import numpy as np


def recall_at_k(
    risk_scores: np.ndarray,
    true_labels: np.ndarray,
    k_fraction: float = 0.10,
) -> float:
    """Fraction of true failures caught within the top-K highest-risk assets."""
    risk_scores = np.asarray(risk_scores)
    true_labels = np.asarray(true_labels)

    n_assets = len(risk_scores)
    total_failures = int(true_labels.sum())

    if total_failures == 0:
        return 0.0

    k = max(1, int(round(k_fraction * n_assets)))
    k = min(k, n_assets)

    top_k_indices = np.argsort(risk_scores)[::-1][:k]
    caught = int(true_labels[top_k_indices].sum())
    return caught / total_failures


def flag_rate(risk_scores: np.ndarray, threshold: float) -> float:
    """Fraction of assets flagged (scored strictly above the threshold)."""
    risk_scores = np.asarray(risk_scores)
    n_assets = len(risk_scores)
    if n_assets == 0:
        return 0.0
    flagged = int((risk_scores > threshold).sum())
    return flagged / n_assets


def within_fp_budget(
    risk_scores: np.ndarray,
    threshold: float,
    budget: float = 0.10,
) -> bool:
    """Is the flag rate within the false-positive budget (flag rate <= budget)?"""
    return flag_rate(risk_scores, threshold) <= budget


def naive_threshold_scores(raw_values: np.ndarray) -> np.ndarray:
    """The naive baseline: raw sensor value as risk score, normalised to [0, 1].

    The dumbest reasonable approach - any real model must beat this baseline's
    recall@K to earn its complexity. Used as the regression benchmark in CI.
    """
    raw_values = np.asarray(raw_values, dtype=float)
    if len(raw_values) == 0:
        return raw_values
    lo, hi = raw_values.min(), raw_values.max()
    if hi == lo:
        return np.zeros_like(raw_values)
    return (raw_values - lo) / (hi - lo)


@dataclass
class GateResult:
    """The verdict of the evaluation gate, with the evidence behind it."""

    passed: bool
    recall: float
    flag_rate_value: float
    baseline_recall: float
    reasons: list[str]


def evaluate_model_gate(
    risk_scores: np.ndarray,
    true_labels: np.ndarray,
    raw_values: np.ndarray,
    threshold: float,
    recall_target: float = 0.80,
    k_fraction: float = 0.10,
    fp_budget: float = 0.10,
) -> GateResult:
    """Decide whether a model passes the P7 evaluation gate.

    PASSES only if ALL three hold:
      1. recall@K >= recall_target
      2. flag rate <= fp_budget
      3. recall@K > naive baseline recall
    """
    model_recall = recall_at_k(risk_scores, true_labels, k_fraction)
    rate = flag_rate(risk_scores, threshold)
    baseline_scores = naive_threshold_scores(raw_values)
    baseline_recall = recall_at_k(baseline_scores, true_labels, k_fraction)

    reasons: list[str] = []
    if model_recall < recall_target:
        reasons.append(f"recall@K {model_recall:.2f} below target {recall_target:.2f}")
    if rate > fp_budget:
        reasons.append(f"flag rate {rate:.2f} exceeds FP budget {fp_budget:.2f}")
    if model_recall <= baseline_recall:
        reasons.append(f"recall@K {model_recall:.2f} does not beat baseline {baseline_recall:.2f}")

    return GateResult(
        passed=(len(reasons) == 0),
        recall=model_recall,
        flag_rate_value=rate,
        baseline_recall=baseline_recall,
        reasons=reasons,
    )
