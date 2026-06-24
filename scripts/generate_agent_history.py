"""
Generate a representative agent-run history for the Power BI capstone (BI-0).

Produces two artifacts:
  data/device_metadata.json  - the asset registry (master / reference data)
  data/agent_history.jsonl   - one real-shaped agent-run record per line

The JSONL records mirror the production agent_audit_log.jsonl schema exactly
(nested schedule_entry, flags list, audit_log list) so that
export_audit_for_bi.py exercises the SAME flatten/derive path it would use on
a real production log. Deterministic given --seed.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

ASSET_TYPES = ["Cylindrical", "Spherical", "Tapered", "Deep-Groove"]
LINES = ["LINE_A", "LINE_B", "LINE_C", "LINE_D"]
CRITICALITIES = ["High", "Medium", "Low"]
CRITICALITY_WEIGHTS = [0.20, 0.50, 0.30]  # ~20% High, 50% Medium, 30% Low

HITL_THRESHOLD = 0.80  # matches the agent: risk > 0.8 -> needs_human (escalate)
MAX_STEPS = 12  # matches the agent step limit
HALT_RATE = 0.02  # ~2% of runs hit the step limit (independent of risk)

# Per-device mean risk by criticality. THIS is the core narrative knob:
# high-criticality assets sit higher on the risk axis, so they cross the 0.8
# HITL line more often -> they become the escalation hotspots the dashboard
# is built to surface.
BASE_RISK = {"High": 0.74, "Medium": 0.54, "Low": 0.36}


def action_for_risk(risk: float) -> str:
    """Map a risk score to a recommended action (low -> high severity)."""
    if risk < 0.30:
        return "Monitor - no action needed"
    if risk < 0.60:
        return "Schedule routine check"
    if risk < 0.80:
        return "Inspect soon - elevated risk"
    return "Immediate inspection - high risk"


def build_device_registry(n_devices: int, rng: np.random.Generator) -> list[dict]:
    """Assign fixed metadata to each device. This is master data."""
    devices = []
    for i in range(1, n_devices + 1):
        crit = str(rng.choice(CRITICALITIES, p=CRITICALITY_WEIGHTS))
        base = BASE_RISK[crit] + rng.normal(0, 0.05)  # within-class spread
        devices.append(
            {
                "device_id": f"BEARING_{i:02d}",
                "asset_type": str(rng.choice(ASSET_TYPES)),
                "line": str(rng.choice(LINES)),
                "criticality": crit,
                "_base_risk": float(np.clip(base, 0.05, 0.78)),
                "_drifts": bool(rng.random() < 0.35),  # 35% are "degrading"
            }
        )
    return devices


def risk_for_run(device: dict, day_frac: float, rng: np.random.Generator) -> float:
    """Draw one run's risk score. Degrading assets accrue an upward time drift;
    every run gets gaussian noise. day_frac in [0,1] = position in the window."""
    risk = device["_base_risk"]
    if device["_drifts"]:
        risk += 0.22 * day_frac  # creeps up across the 90-day window
    risk += rng.normal(0, 0.13)  # per-run noise
    return float(np.clip(risk, 0.0, 1.0))


def make_audit_log(start: datetime, n_steps: int) -> list[str]:
    """audit_log entries: 'ISO-8601 | step ...'. Entry [0] marks the run start
    (this is the timestamp source the exporter relies on under option B)."""
    lines, t = [], start
    for s in range(1, n_steps + 1):
        lines.append(f"{t.isoformat()} | step {s} executed")
        t += timedelta(seconds=2 + 3 * s)
    return lines


def build_run_record(
    device: dict, start: datetime, day_frac: float, rng: np.random.Generator
) -> dict:
    risk = risk_for_run(device, day_frac, rng)
    action = action_for_risk(risk)

    halted = rng.random() < HALT_RATE
    escalated = (not halted) and (risk > HITL_THRESHOLD)

    if halted:
        status, steps, flags = "halted_step_limit", MAX_STEPS, ["halted_step_limit"]
    elif escalated:
        status, steps = "needs_human", int(rng.integers(4, 8))
        flags = ["escalated", "HUMAN_REVIEW_REQUIRED"]
    else:
        status, steps, flags = "done", int(rng.integers(3, 7)), []

    audit_log = make_audit_log(start, steps)

    if status == "done":
        schedule_entry = {
            "device_id": device["device_id"],
            "action": action,
            "risk_score": round(risk, 4),
            "created_at": start.isoformat(),
        }
    else:
        schedule_entry = None

    return {
        "device_id": device["device_id"],
        "risk_score": round(risk, 4),
        "recommended_action": action,
        "status": status,
        "flags": flags,
        "steps_taken": steps,
        "explanation": (
            f"Device {device['device_id']} has a risk score of {risk:.2f}. "
            f"Recommended action: {action}."
        ),
        "schedule_entry": schedule_entry,
        "audit_log": audit_log,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Generate representative agent history (BI-0).")
    p.add_argument("--devices", type=int, default=50)
    p.add_argument("--runs", type=int, default=2000)
    p.add_argument("--days", type=int, default=90)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", type=Path, default=Path("data"))
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    devices = build_device_registry(args.devices, rng)

    end = datetime.now(timezone.utc).replace(microsecond=0)
    start_window = end - timedelta(days=args.days)
    window_seconds = args.days * 24 * 3600

    records = []
    for _ in range(args.runs):
        device = devices[int(rng.integers(0, len(devices)))]
        offset = float(rng.random()) * window_seconds
        run_start = start_window + timedelta(seconds=offset)
        records.append(build_run_record(device, run_start, offset / window_seconds, rng))

    records.sort(key=lambda r: r["audit_log"][0])  # ISO prefix sorts chronologically

    registry_path = args.out_dir / "device_metadata.json"
    public = [{k: v for k, v in d.items() if not k.startswith("_")} for d in devices]
    registry_path.write_text(json.dumps(public, indent=2))

    history_path = args.out_dir / "agent_history.jsonl"
    with history_path.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")

    escalated = sum(1 for r in records if "escalated" in r["flags"])
    halted = sum(1 for r in records if r["status"] == "halted_step_limit")
    print(f"Devices:   {len(devices)}")
    print(f"Runs:      {len(records)}")
    print(f"Escalated: {escalated} ({escalated / len(records):.1%})")
    print(f"Halted:    {halted} ({halted / len(records):.1%})")
    print(f"Window:    {start_window.date()} -> {end.date()}")
    print(f"Wrote:     {registry_path}")
    print(f"Wrote:     {history_path}")


if __name__ == "__main__":
    main()
