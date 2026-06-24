"""
Run the compiled LangGraph maintenance agent and print its flow + audit trail.

Run:  python scripts/run_graph_demo.py
"""

import joblib

from src.agents.graph_agent import run_graph_agent

bundle = joblib.load("models/anomaly_model.joblib")


def show(title: str, readings: list[float]) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)
    final = run_graph_agent("BEARING_07", readings, bundle)
    print(f"\nFinal status:   {final['status']}")
    print(f"Risk score:     {final['risk_score']}")
    print(f"Flags:          {final['flags']}")
    if final.get("explanation"):
        print(f"Explanation:    {final['explanation'][:110]}...")
    if final.get("schedule_entry"):
        print(f"Scheduled:      {final['schedule_entry']['action']}")
    print("\n--- AUDIT LOG (accumulated via the add reducer) ---")
    for entry in final["audit_log"]:
        print(f"  {entry}")


show("CASE 1: Normal readings", [10.1, 9.9, 10.2, 9.8, 10.0, 10.1, 9.9, 10.0])
show("CASE 2: Degrading readings", [10.1, 10.3, 9.8, 11.2, 14.5, 18.9, 22.0, 25.0])
