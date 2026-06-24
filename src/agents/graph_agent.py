"""
Assemble the maintenance agent as a compiled LangGraph.

Graph shape:
    START -> score -> [route_after_score]
                        - "escalate" -> END (paused, needs_human)   [HITL]
                        - "continue" -> explain -> schedule -> END

This is the production form of the hand-rolled loop: nodes = tools (doing),
conditional edge = the router (deciding). Same governance, framework-driven.
"""

from langgraph.graph import END, START, StateGraph

from src.agents.graph_nodes import (
    explain_node,
    make_score_node,
    schedule_node,
)
from src.agents.graph_state import GraphState


def route_after_score(state: GraphState) -> str:
    """After scoring: HIGH risk -> escalate (HITL pause); else -> continue."""
    if state["risk_score"] > 0.8:
        return "escalate"
    else:
        return "continue"


def escalate_node(state: GraphState) -> dict:
    """High-risk path: flag for human review and stop (do NOT auto-schedule)."""
    from datetime import datetime, timezone

    ts = datetime.now(timezone.utc).isoformat()
    return {
        "status": "needs_human",
        "flags": ["HUMAN_REVIEW_REQUIRED"],
        "audit_log": [f"{ts} | escalate_node: risk>0.8, paused for human review"],
    }


def build_agent_graph(bundle: dict):
    """Build and compile the maintenance agent graph."""
    graph = StateGraph(GraphState)

    # Nodes (the doing)
    graph.add_node("score", make_score_node(bundle))
    graph.add_node("explain", explain_node)
    graph.add_node("schedule", schedule_node)
    graph.add_node("escalate", escalate_node)

    # Edges (the flow)
    graph.add_edge(START, "score")
    graph.add_conditional_edges(
        "score",
        route_after_score,
        {"escalate": "escalate", "continue": "explain"},
    )
    graph.add_edge("explain", "schedule")
    graph.add_edge("schedule", END)
    graph.add_edge("escalate", END)

    return graph.compile()


def run_graph_agent(device_id: str, readings: list[float], bundle: dict) -> dict:
    """Run the compiled graph and return the final state."""
    app = build_agent_graph(bundle)
    initial: GraphState = {
        "device_id": device_id,
        "readings": readings,
        "risk_score": None,
        "top_sensors": [],
        "recommended_action": None,
        "explanation": None,
        "schedule_entry": None,
        "flags": [],
        "audit_log": [f"graph: started for {device_id}"],
        "status": "running",
    }
    return app.invoke(initial)
