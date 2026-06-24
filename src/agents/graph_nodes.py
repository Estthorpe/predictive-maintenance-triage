"""
LangGraph nodes for the maintenance agent.

Each node is a thin adapter: it reads from the state dict, calls the SAME
logic the hand-rolled tools used (reused, not rewritten), and returns a dict
of updates. LangGraph merges those updates into the state (audit_log/flags
accumulate via their reducers).
"""

from datetime import datetime, timezone

import pandas as pd

from src.agents.graph_state import GraphState
from src.features.engineering import compute_baseline_zscore, compute_features
from src.genai.report import generate_report

HITL_RISK_THRESHOLD = 0.8


def _ts(msg: str) -> str:
    return f"{datetime.now(timezone.utc).isoformat()} | {msg}"


def make_score_node(bundle: dict):
    """Build the score node (closure injects the model bundle)."""

    def score_node(state: GraphState) -> dict:
        values = pd.Series(state["readings"])
        features = compute_features(values, window=bundle["window"])
        features["baseline_zscore"] = compute_baseline_zscore(
            values, bundle["baseline_mean"], bundle["baseline_std"]
        ).to_numpy()

        model_features = features[bundle["feature_names"]]
        scores = bundle["model"].score(model_features)
        latest_raw = float(scores[-1])
        risk = round(1.0 / (1.0 + pow(2.718281828, -latest_raw)), 4)

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

        flags = ["CRITICAL"] if risk > HITL_RISK_THRESHOLD else []
        return {
            "risk_score": risk,
            "top_sensors": top,
            "recommended_action": action,
            "flags": flags,
            "audit_log": [_ts(f"score_node: risk={risk}, action='{action}'")],
        }

    return score_node


def explain_node(state: GraphState) -> dict:
    report = generate_report(
        device_id=state["device_id"],
        risk_score=state["risk_score"] or 0.0,
        top_sensors=state["top_sensors"],
        recommended_action=state["recommended_action"] or "",
    )
    return {
        "explanation": report["summary"],
        "audit_log": [_ts(f"explain_node: source={report['source']}")],
    }


def schedule_node(state: GraphState) -> dict:
    entry = {
        "device_id": state["device_id"],
        "action": state["recommended_action"],
        "risk_score": state["risk_score"],
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    return {
        "schedule_entry": entry,
        "status": "done",
        "audit_log": [_ts(f"schedule_node: scheduled '{state['recommended_action']}'")],
    }
