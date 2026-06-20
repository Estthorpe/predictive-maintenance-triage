# Model Card - P7 Anomaly Detection

## Models
- Isolation Forest (unsupervised) - primary detector, trained on healthy data.
- LightGBM (supervised) - evaluated as an alternative; needs labelled anomalies
  in training.

## Intended use
Per-asset failure-risk scoring for manufacturing/IoT predictive maintenance.
Scores feed the serving API and the Maintenance Scheduler Agent.

## Evaluation
Judged by the Phase-2 gate: recall@K (K = top 10% of fleet), false-positive
budget (<= 10% flagged), and a beat-naive-baseline regression check.

## Honest performance summary
- Real clear failures (NASA IMS Bearing 3): recall@K 1.00.
- Real subtle failures (Bearing 4): recall@K 0.48 -> 0.87 with the per-asset
  baseline z-score feature.
- On a bias-free time-based evaluation, a calibrated raw-RMS threshold remains a
  strong baseline, because vibration magnitude is physically predictive of
  end-of-life. See docs/phase3_model_findings.md for the full investigation.

## Limitations
- Isolation Forest detects spatial outliers, not temporal trajectories; subtle,
  scattered degradation is hard for it.
- Heuristic labels (control-chart / time-based) are not gold-standard ground
  truth and can bias evaluation.

## Future work
Sequence models (LSTM/TCN) for trajectory-aware detection; improved labelling.
