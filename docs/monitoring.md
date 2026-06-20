# Monitoring & Observability - P7

## Two planes of monitoring
1. Operational (is the SERVICE healthy?): Prometheus metrics at /metrics -
   request count, latency histogram, error count.
2. Model validity (is the MODEL still valid?): data drift detection
   (src/monitoring/drift.py) - KS statistic of live features vs training baseline.

## Drift detection
- compute_drift(reference, current) -> KS statistic (0 = identical, 1 = fully
  different).
- is_drifted(score, threshold=0.2) -> True when drift exceeds the threshold.

## Why two planes
A service can be perfectly healthy (fast, no errors) while the MODEL is silently
wrong because the world drifted. Operational metrics catch service problems;
drift detection catches model-validity problems. Both are needed.

## Response
See docs/runbook.md for alert thresholds, incident response, the retraining
trigger, and escalation paths.
