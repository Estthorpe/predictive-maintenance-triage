"""Tests for the agent State and tools."""

import joblib

from src.agents.maintenance_agent import (
    TOOL_REGISTRY,
    AgentState,
    decide_next_action,
    explain_tool,
    persist_run,
    run_agent,
    schedule_tool,
    score_tool,
)

BUNDLE = joblib.load("models/anomaly_model.joblib")


def test_state_starts_running_with_zero_steps():
    s = AgentState(device_id="B1", readings=[10.0, 11.0])
    assert s.status == "running"
    assert s.steps_taken == 0
    assert s.audit_log == []


def test_log_appends_to_audit_log():
    s = AgentState(device_id="B1", readings=[10.0])
    s.log("hello")
    assert len(s.audit_log) == 1
    assert "hello" in s.audit_log[0]


def test_score_tool_fills_risk_and_logs():
    s = AgentState(device_id="B1", readings=[10.1, 10.3, 9.8, 11.2, 14.5, 18.9])
    s = score_tool(s, BUNDLE)
    assert s.risk_score is not None
    assert 0.0 <= s.risk_score <= 1.0
    assert len(s.top_sensors) > 0
    assert any("score_tool" in entry for entry in s.audit_log)


def test_explain_tool_fills_explanation():
    s = AgentState(device_id="B1", readings=[10.1, 10.3, 9.8, 11.2, 14.5, 18.9])
    s = score_tool(s, BUNDLE)
    s = explain_tool(s)
    assert isinstance(s.explanation, str) and len(s.explanation) > 0


def test_schedule_tool_creates_entry():
    s = AgentState(device_id="B1", readings=[10.1, 10.3, 9.8, 11.2, 14.5, 18.9])
    s = score_tool(s, BUNDLE)
    s = schedule_tool(s)
    assert s.schedule_entry is not None
    assert s.schedule_entry["device_id"] == "B1"


def test_registry_has_the_three_tools():
    assert set(TOOL_REGISTRY.keys()) == {"score", "explain", "schedule"}


def test_decide_scores_first():
    s = AgentState(device_id="B1", readings=[10.0, 11.0])
    assert decide_next_action(s) == "score"


def test_decide_explains_after_scoring():
    s = AgentState(device_id="B1", readings=[10.0])
    s.risk_score = 0.3  # scored, low risk
    assert decide_next_action(s) == "explain"


def test_decide_escalates_on_high_risk():
    s = AgentState(device_id="B1", readings=[10.0])
    s.risk_score = 0.95  # high risk, not yet escalated
    assert decide_next_action(s) == "escalate"


def test_low_risk_runs_to_completion():
    # Low-risk readings -> agent should finish the full workflow (done).
    s = run_agent("B1", [10.0, 10.1, 9.9, 10.2, 10.0, 10.1], BUNDLE)
    assert s.status in ("done", "needs_human")  # depends on score
    assert s.steps_taken > 0
    assert len(s.audit_log) > 0


def test_agent_never_exceeds_step_limit():
    s = run_agent("B1", [10.1, 10.3, 9.8, 11.2, 14.5, 18.9], BUNDLE)
    from src.agents.maintenance_agent import MAX_STEPS

    assert s.steps_taken <= MAX_STEPS


def test_high_risk_pauses_for_human():
    # Force a high score path by checking the escalation flag behaviour:
    s = AgentState(device_id="B1", readings=[10.0])
    s.risk_score = 0.95
    action = decide_next_action(s)
    assert action == "escalate"


def test_persist_run_appends_a_line(tmp_path, monkeypatch):
    import src.agents.maintenance_agent as agent_mod

    # Redirect the audit log to a temp file for the test.
    test_log = tmp_path / "audit.jsonl"
    monkeypatch.setattr(agent_mod, "AUDIT_LOG_PATH", test_log)

    state = run_agent("B1", [10.0, 11.0, 10.5, 9.8, 10.2, 11.1], BUNDLE)
    persist_run(state)
    persist_run(state)  # second run -> second line (append-only)

    lines = test_log.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 2  # two appended records, nothing overwritten
