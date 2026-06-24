"""Tests for the LangGraph agent."""

import joblib

from src.agents.graph_agent import route_after_score, run_graph_agent

BUNDLE = joblib.load("models/anomaly_model.joblib")


def test_router_escalates_high_risk():
    assert route_after_score({"risk_score": 0.95}) == "escalate"


def test_router_continues_low_risk():
    assert route_after_score({"risk_score": 0.3}) == "continue"


def test_graph_runs_to_completion():
    final = run_graph_agent("B1", [10.1, 10.3, 9.8, 11.2, 14.5, 18.9], BUNDLE)
    assert final["risk_score"] is not None
    assert final["status"] in ("done", "needs_human")
    assert len(final["audit_log"]) > 1  # accumulated across nodes via reducer


def test_graph_audit_log_accumulates():
    # The add reducer must accumulate entries from multiple nodes.
    final = run_graph_agent("B1", [10.0, 10.1, 9.9, 10.2, 10.0, 10.1], BUNDLE)
    # started + score + (explain + schedule) OR (escalate) -> several entries
    assert any("score_node" in e for e in final["audit_log"])
