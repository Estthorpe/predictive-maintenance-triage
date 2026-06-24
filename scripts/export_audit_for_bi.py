"""
Flatten the agent audit grain into Power BI-ready tables (BI-0).

Reads:
  data/agent_history.jsonl   (or any real-shaped audit JSONL via --input)
  data/device_metadata.json  (the asset registry / master data)

Writes:
  data/bi/fct_triage.csv     (one row per agent run; the fact table)
  data/bi/dim_device.csv     (one row per device; the device dimension)

Key responsibilities:
  * derive run_timestamp from audit_log[0]  (option B: always the run start)
  * derive escalated (bool) from the flags list
  * drop nested lists/objects so the fact stays a clean rectangle
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def derive_run_timestamp(record: dict) -> str:
    """Option B: parse the timestamp from audit_log[0] for EVERY run.

    Each audit_log entry is 'ISO-8601 | step ...'. The first entry marks the
    run start. Escalated/halted runs have schedule_entry=null, so audit_log[0]
    is the only timestamp source guaranteed to be present.
    """
    audit_log = record.get("audit_log") or []
    if not audit_log:
        raise ValueError(f"Empty audit_log for device {record.get('device_id')!r}")
    return audit_log[0].split("|", 1)[0].strip()  # ISO-8601; parsed on load


def flatten_record(record: dict) -> dict:
    """Collapse one nested audit record into one flat fact row."""
    flags = record.get("flags") or []
    return {
        "device_id": record["device_id"],
        "run_timestamp": derive_run_timestamp(record),
        "risk_score": record["risk_score"],
        "status": record["status"],
        "recommended_action": record["recommended_action"],
        "steps_taken": record["steps_taken"],
        "escalated": "escalated" in flags,
        "human_review_required": "HUMAN_REVIEW_REQUIRED" in flags,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Export agent audit log to BI tables (BI-0).")
    p.add_argument("--input", type=Path, default=Path("data/agent_history.jsonl"))
    p.add_argument("--registry", type=Path, default=Path("data/device_metadata.json"))
    p.add_argument("--out-dir", type=Path, default=Path("data/bi"))
    args = p.parse_args()

    rows = []
    with args.input.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(flatten_record(json.loads(line)))

    fct = pd.DataFrame(rows)
    fct["run_timestamp"] = pd.to_datetime(fct["run_timestamp"], utc=True)
    fct = fct.sort_values("run_timestamp").reset_index(drop=True)
    fct.insert(0, "run_id", range(1, len(fct) + 1))  # surrogate key for the grain

    devices = json.loads(args.registry.read_text())
    dim_device = pd.DataFrame(devices)[["device_id", "asset_type", "line", "criticality"]]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    fct_path = args.out_dir / "fct_triage.csv"
    dim_path = args.out_dir / "dim_device.csv"
    fct.to_csv(fct_path, index=False)
    dim_device.to_csv(dim_path, index=False)

    print("fct_triage.csv")
    print(f"  rows:            {len(fct)}")
    print(
        f"  date span:       {fct['run_timestamp'].min().date()} -> {fct['run_timestamp'].max().date()}"
    )
    print(f"  escalation %:    {fct['escalated'].mean():.1%}")
    print(f"  null timestamps: {fct['run_timestamp'].isna().sum()}")
    print(f"  status mix:      {fct['status'].value_counts().to_dict()}")
    print(f"  action mix:      {fct['recommended_action'].value_counts().to_dict()}")
    print("dim_device.csv")
    print(f"  devices:         {len(dim_device)}")
    print(f"  criticality:     {dim_device['criticality'].value_counts().to_dict()}")
    print(f"Wrote: {fct_path}")
    print(f"Wrote: {dim_path}")


if __name__ == "__main__":
    main()
