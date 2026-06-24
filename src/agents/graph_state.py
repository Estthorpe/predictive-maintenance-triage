"""
LangGraph rebuild of the maintenance agent: STATE.

The production version of the hand-rolled AgentState. LangGraph flows state
between nodes as a TypedDict: each node returns a dict of updates, and the
framework merges them in.

The audit_log uses an `add` reducer (Annotated[..., operator.add]) so that
entries from every node ACCUMULATE rather than overwrite - this is how the
append-only audit log works across nodes, declaratively.
"""

import operator
from typing import Annotated, TypedDict


class GraphState(TypedDict):
    # --- Input ---
    device_id: str
    readings: list[float]

    # --- Filled by the score node ---
    risk_score: float | None
    top_sensors: list[str]
    recommended_action: str | None

    # --- Filled by the explain node ---
    explanation: str | None

    # --- Filled by the schedule node ---
    schedule_entry: dict | None

    # --- Governance ---
    flags: Annotated[list[str], operator.add]  # accumulates across nodes
    audit_log: Annotated[list[str], operator.add]  # append-only via reducer
    status: str
