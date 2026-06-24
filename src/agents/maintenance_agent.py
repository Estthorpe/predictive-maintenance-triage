"""
Hand-rolled maintenance agent: State + Tools (foundation).

STATE = the agent''s "clipboard" - the single object carried through the loop,
remembering the goal, what''s been done, results, flags, audit log, and step
count.

TOOL REGISTRY = the constrained set of functions the agent is allowed to call
(safety structure #1). Each tool wraps work built in earlier phases and appends
to the audit log (accountability from line one). The agent cannot act outside
this toolbox.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.genai.report import generate_report


@dataclass
class AgentState:
    """The agent''s clipboard: carried through the loop, updated each step."""

    # --- Input ---
    device_id: str
    readings: list[float]

    # --- Filled by the scoring tool ---
    risk_score: float | None = None
    top_sensors: list[str] = field(default_factory=list)
    recommended_action: str | None = None

    # --- Filled by the explain tool ---
    explanation: str | None = None

    # --- Filled by the schedule tool ---
    schedule_entry: dict | None = None

    # --- Safety / governance ---
    flags: list[str] = field(default_factory=list)
    audit_log: list[str] = field(default_factory=list)
    steps_taken: int = 0
    status: str = "running"

    def log(self, message: str) -> None:
        """Append a timestamped entry to the append-only audit log."""
        ts = datetime.now(timezone.utc).isoformat()
        self.audit_log.append(f"{ts} | {message}")


# --- Tool registry -----------------------------------------------------------
# Each tool takes the state, does ONE job, updates the state, logs it, returns
# the state. The agent may only use these tools.


def score_tool(state: AgentState, bundle: dict) -> AgentState:
    """Score the readings with the anomaly model (wraps the prediction logic)."""
    from src.features.engineering import compute_baseline_zscore, compute_features

    values = pd.Series(state.readings)
    features = compute_features(values, window=bundle["window"])
    features["baseline_zscore"] = compute_baseline_zscore(
        values, bundle["baseline_mean"], bundle["baseline_std"]
    ).to_numpy()

    model_features = features[bundle["feature_names"]]
    scores = bundle["model"].score(model_features)
    latest_raw = float(scores[-1])
    risk = 1.0 / (1.0 + pow(2.718281828, -latest_raw))

    latest = features.iloc[-1].sort_values(ascending=False)
    top = list(latest.head(3).index)

    if risk >= 0.75:
        action = "URGENT: schedule immediate inspection"
    elif risk >= 0.50:
        action = "Inspect soon - elevated risk"
    elif risk >= 0.25:
        action = "Monitor - mild elevation"
    else:
        action = "No action - within normal range"

    state.risk_score = round(risk, 4)
    state.top_sensors = top
    state.recommended_action = action
    state.log(f"score_tool: risk={state.risk_score}, action='{action}'")
    return state


def explain_tool(state: AgentState) -> AgentState:
    """Generate a grounded explanation (wraps generate_report)."""
    report = generate_report(
        device_id=state.device_id,
        risk_score=state.risk_score or 0.0,
        top_sensors=state.top_sensors,
        recommended_action=state.recommended_action or "",
    )
    state.explanation = report["summary"]
    state.log(f"explain_tool: source={report['source']}")
    return state


def schedule_tool(state: AgentState) -> AgentState:
    """Produce a maintenance schedule entry (the agent''s ACTION)."""
    entry = {
        "device_id": state.device_id,
        "action": state.recommended_action,
        "risk_score": state.risk_score,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    state.schedule_entry = entry
    state.log(f"schedule_tool: scheduled '{state.recommended_action}'")
    return state


# The registry: the ONLY tools the agent may use.
TOOL_REGISTRY = {
    "score": score_tool,
    "explain": explain_tool,
    "schedule": schedule_tool,
}


# --- The agent loop ----------------------------------------------------------

MAX_STEPS = 12  # safety structure #5: hard runaway limit
HITL_RISK_THRESHOLD = 0.8  # risk above this needs human sign-off


def decide_next_action(state: AgentState) -> str:
    """Rule-based router: given the state, what should the agent do next?

    Returns one of: "score", "escalate", "explain", "schedule", "done".
    Order matters: score -> (maybe escalate) -> explain -> schedule -> done.
    """
    if state.risk_score is None:
        return "score"
    elif state.risk_score > HITL_RISK_THRESHOLD and "escalated" not in state.flags:
        return "escalate"
    elif state.explanation is None:
        return "explain"
    elif state.schedule_entry is None:
        return "schedule"
    else:
        return "done"


def run_agent(device_id: str, readings: list[float], bundle: dict) -> AgentState:
    """Run the governed maintenance-triage agent loop.

    observe -> decide -> act -> log, until done or a stop condition fires.
    Safety: step limit (runaway), HITL escalation (authority), audit log
    (accountability) are all enforced inside this loop.
    """
    state = AgentState(device_id=device_id, readings=readings)
    state.log(f"agent: started for {device_id} with {len(readings)} readings")

    while state.status == "running":
        # Safety structure #5: hard step limit (cannot run forever).
        if state.steps_taken >= MAX_STEPS:
            state.status = "halted_step_limit"
            state.log(f"agent: halted - reached step limit ({MAX_STEPS})")
            break

        state.steps_taken += 1
        action = decide_next_action(state)
        state.log(f"agent: step {state.steps_taken} -> decided '{action}'")

        if action == "score":
            state = score_tool(state, bundle)

        elif action == "escalate":
            # Human-in-the-loop: flag and PAUSE. The agent does not proceed to
            # act on a high-risk case without human sign-off.
            state.flags.append("escalated")
            state.flags.append("HUMAN_REVIEW_REQUIRED")
            state.status = "needs_human"
            state.log(
                f"agent: HITL - risk {state.risk_score} > {HITL_RISK_THRESHOLD}, "
                "paused for human review"
            )
            break

        elif action == "explain":
            state = explain_tool(state)

        elif action == "schedule":
            state = schedule_tool(state)

        elif action == "done":
            state.status = "done"
            state.log("agent: workflow complete")
            break

    return state


AUDIT_LOG_PATH = Path("data") / "agent_audit_log.jsonl"


def persist_run(state: AgentState) -> None:
    """Append one agent run to the append-only audit log (JSON-lines).

    Append-only: each run adds exactly one line; nothing is ever overwritten,
    so the record is tamper-evident. This file is the analytics "grain" - each
    line becomes a row in downstream reporting (e.g. Power BI).
    """
    AUDIT_LOG_PATH.parent.mkdir(exist_ok=True)
    record = {
        "device_id": state.device_id,
        "risk_score": state.risk_score,
        "recommended_action": state.recommended_action,
        "status": state.status,
        "flags": state.flags,
        "steps_taken": state.steps_taken,
        "explanation": state.explanation,
        "schedule_entry": state.schedule_entry,
        "audit_log": state.audit_log,
    }
    with open(AUDIT_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
