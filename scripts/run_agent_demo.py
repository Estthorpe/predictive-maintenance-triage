"""
Run the governed maintenance agent end-to-end and print its audit trail.

Shows the agent walking the triage workflow (score -> explain -> schedule) and,
on a high-risk case, PAUSING for human review (HITL).

Run:  python scripts/run_agent_demo.py
"""

import joblib

from src.agents.maintenance_agent import run_agent

bundle = joblib.load("models/anomaly_model.joblib")


def show(title: str, readings: list[float]) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)
    state = run_agent("BEARING_07", readings, bundle)

    print(f"\nFinal status:   {state.status}")
    print(f"Risk score:     {state.risk_score}")
    print(f"Steps taken:    {state.steps_taken}")
    print(f"Flags:          {state.flags}")
    if state.explanation:
        print(f"Explanation:    {state.explanation[:120]}...")
    if state.schedule_entry:
        print(f"Scheduled:      {state.schedule_entry['action']}")
    print("\n--- AUDIT LOG (append-only) ---")
    for entry in state.audit_log:
        print(f"  {entry}")


# Case 1: normal readings -> should run the full workflow to "done"
show(
    "CASE 1: Normal readings (expect full workflow -> done)",
    [10.1, 9.9, 10.2, 9.8, 10.0, 10.1, 9.9, 10.0],
)

# Case 2: spiky readings -> higher risk (may escalate to human depending on score)
show("CASE 2: Degrading readings (higher risk)", [10.1, 10.3, 9.8, 11.2, 14.5, 18.9, 22.0, 25.0])
