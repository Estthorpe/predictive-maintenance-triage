# P7 Runbook - Predictive Maintenance API

Operational guide for responding to alerts from the monitoring layer. Monitoring
is only useful if an alarm leads to action: this document is the "what to do"
half of detector + threshold + runbook.

## System at a glance
- Service: FastAPI app (`src/serving/app.py`), containerised (Dockerfile).
- Model: Isolation Forest anomaly detector, loaded once at startup from
  `models/anomaly_model.joblib` (bundle: model + healthy baseline + config).
- Endpoints: `/health` (liveness), `/ready` (model loaded), `/predict_failure`
  (scoring), `/metrics` (Prometheus).
- Drift detection: `src/monitoring/drift.py` (KS statistic vs training baseline).

## Metrics to watch (Prometheus)
- `p7_predictions_total`        - request volume (throughput).
- `p7_prediction_errors_total`  - failed requests.
- `p7_prediction_latency_seconds` - latency distribution.
- Drift score (KS statistic) of live features vs the training baseline.

## Alert thresholds
| Alert            | Condition                                  | Severity |
|------------------|--------------------------------------------|----------|
| Service down     | `/health` not 200 for > 1 min              | Critical |
| Not ready        | `/ready` not "ready" for > 2 min           | High     |
| High error rate  | errors / requests > 5% over 5 min          | High     |
| High latency     | p95 latency > 1s over 5 min                | Medium   |
| Data drift       | drift score > 0.2 (DEFAULT_DRIFT_THRESHOLD)| High     |

## Incident response

### 1. Service down (`/health` failing)
- Check the container/process is running (`docker ps`).
- Check logs for a crash on startup.
- Restart the service; if it crashes again, roll back to the last good image.

### 2. Not ready (`/ready` reports model not loaded)
- The model artifact is missing or corrupt at `models/anomaly_model.joblib`.
- Regenerate it: `python scripts/train_and_save_model.py`, then restart.
- In the container, the artifact is built at image build time - rebuild the image.

### 3. High error rate
- Inspect recent failed requests. Common cause: malformed input that slipped
  past validation, or an unexpected feature shape.
- The Pydantic contract returns 422 for bad input by design - a spike in 422s
  means a CLIENT is sending bad data; a spike in 500s means a SERVER bug.

### 4. High latency
- Check for cold-start (first request after deploy is slow - expected).
- Sustained high latency: check host CPU/memory; consider scaling out.

### 5. Data drift (the model-validity alarm)
- A drift score > threshold means live data no longer matches training data.
- Diagnose: is it a real fleet-wide change (e.g. seasonal), a sensor fault, or
  a new asset type? Check which feature drifted and by how much.
- Short term: keep serving but flag predictions as lower-confidence; notify
  the maintenance team that scores may be unreliable.
- Long term: RETRAIN (see below).

## Retraining trigger (explicit rule)
Retrain the model when ANY of the following holds:
- Drift score > 0.2 on the primary features for 3 consecutive monitoring windows
  (sustained drift, not a one-off blip), OR
- Error rate or false-alarm complaints from operators rise materially, OR
- A scheduled cadence elapses (e.g. quarterly) as a safety net.

Retraining procedure (currently manual / simulated):
1. Collect a fresh labelled/representative dataset reflecting current conditions.
2. Re-run the training pipeline: `python scripts/train_and_save_model.py`.
3. Evaluate the new model through the Phase-2 gate (recall@K, FP budget,
   beat-baseline) BEFORE promoting it - a new model must pass the same gate.
4. Swap the artifact and restart; monitor drift + metrics post-deploy.

## Escalation
- Critical (service down) -> page on-call immediately.
- High (drift, error rate) -> notify the ML on-call within the hour.
- Medium (latency) -> ticket for next business day.
