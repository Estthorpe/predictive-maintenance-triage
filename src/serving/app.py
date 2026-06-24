"""
FastAPI serving app for P7 predictive maintenance.

Model artifact loaded once at startup (lifespan handler) and reused per request.
Endpoints:
  GET  /health          - is the service alive?
  GET  /ready           - is the model loaded and ready?
  POST /predict_failure - score a window of sensor readings for one asset.
  GET  /metrics         - Prometheus operational metrics (requests, latency, errors).

No training-serving skew: same compute_features as training; z-score uses the
baseline SAVED IN THE BUNDLE.
"""

import time
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, Field

from src.agents.maintenance_agent import persist_run, run_agent
from src.features.engineering import compute_baseline_zscore, compute_features
from src.genai.report import generate_report

MODEL_PATH = Path("models") / "anomaly_model.joblib"

# Load .env so ANTHROPIC_API_KEY is available to the LLM report generator.
load_dotenv()

# --- Prometheus metrics (the operational vitals) ---
PREDICTIONS_TOTAL = Counter("p7_predictions_total", "Total prediction requests served.")
PREDICTION_ERRORS = Counter("p7_prediction_errors_total", "Total prediction requests that errored.")
PREDICTION_LATENCY = Histogram("p7_prediction_latency_seconds", "Time spent serving a prediction.")

_bundle: dict | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _bundle
    if MODEL_PATH.exists():
        _bundle = joblib.load(MODEL_PATH)
    yield


app = FastAPI(title="P7 Predictive Maintenance API", version="0.1.0", lifespan=lifespan)


class PredictRequest(BaseModel):
    device_id: str = Field(..., min_length=1, examples=["BEARING_07"])
    values: list[float] = Field(
        ...,
        min_length=1,
        description="Recent sensor values, oldest first.",
        examples=[[10.1, 10.3, 9.8, 11.2, 14.5, 18.9]],
    )


class PredictResponse(BaseModel):
    device_id: str
    risk_score: float
    top_sensors: list[str]
    recommended_action: str


def _recommended_action(risk: float) -> str:
    if risk >= 0.75:
        return "URGENT: schedule immediate inspection"
    if risk >= 0.50:
        return "Inspect soon - elevated risk"
    if risk >= 0.25:
        return "Monitor - mild elevation"
    return "No action - within normal range"


@app.get("/health")
def health() -> dict:
    return {"status": "alive"}


@app.get("/ready")
def ready() -> dict:
    if _bundle is None:
        return {"status": "not_ready", "reason": "model not loaded"}
    return {"status": "ready", "features": len(_bundle["feature_names"])}


@app.get("/metrics")
def metrics() -> Response:
    """Prometheus scrape target: operational metrics in Prometheus text format."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict_failure", response_model=PredictResponse)
def predict_failure(req: PredictRequest) -> PredictResponse:
    if _bundle is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    PREDICTIONS_TOTAL.inc()  # count this request
    start = time.perf_counter()
    try:
        values = pd.Series(req.values)

        features = compute_features(values, window=_bundle["window"])
        features["baseline_zscore"] = compute_baseline_zscore(
            values,
            baseline_mean=_bundle["baseline_mean"],
            baseline_std=_bundle["baseline_std"],
        ).to_numpy()

        model_features = features[_bundle["feature_names"]]
        scores = _bundle["model"].score(model_features)
        latest_raw = float(scores[-1])
        risk = 1.0 / (1.0 + pow(2.718281828, -latest_raw))

        latest = features.iloc[-1].sort_values(ascending=False)
        top_sensors = list(latest.head(3).index)

        return PredictResponse(
            device_id=req.device_id,
            risk_score=round(risk, 4),
            top_sensors=top_sensors,
            recommended_action=_recommended_action(risk),
        )
    except Exception:
        PREDICTION_ERRORS.inc()  # count failures
        raise
    finally:
        PREDICTION_LATENCY.observe(time.perf_counter() - start)


class ExplainResponse(BaseModel):
    device_id: str
    risk_score: float
    summary: str
    source: str  # "llm" or "fallback"
    prompt_version: str


@app.post("/explain", response_model=ExplainResponse)
def explain(req: PredictRequest) -> ExplainResponse:
    """Grounded narrative report: the MODEL scores, the LLM explains.

    Reuses the exact prediction logic, then passes the model''s REAL output to
    the report generator. The LLM never decides the risk - it only translates.
    """
    if _bundle is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # 1. The model decides (authority) - reuse the prediction path.
    prediction = predict_failure(req)

    # 2. The LLM explains (presentation) - grounded in the model''s real output.
    report = generate_report(
        device_id=prediction.device_id,
        risk_score=prediction.risk_score,
        top_sensors=prediction.top_sensors,
        recommended_action=prediction.recommended_action,
    )

    return ExplainResponse(
        device_id=prediction.device_id,
        risk_score=prediction.risk_score,
        summary=report["summary"],
        source=report["source"],
        prompt_version=report["prompt_version"],
    )


class TriageResponse(BaseModel):
    device_id: str
    risk_score: float | None
    status: str
    flags: list[str]
    recommended_action: str | None
    explanation: str | None
    steps_taken: int
    audit_log: list[str]


@app.post("/triage", response_model=TriageResponse)
def triage(req: PredictRequest) -> TriageResponse:
    """Run the governed maintenance agent on a window of readings.

    The agent orchestrates score -> explain -> schedule, escalates high-risk
    cases for human review, and the full run is persisted to the append-only
    audit log.
    """
    if _bundle is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    state = run_agent(req.device_id, req.values, _bundle)
    persist_run(state)  # append-only accountability

    return TriageResponse(
        device_id=state.device_id,
        risk_score=state.risk_score,
        status=state.status,
        flags=state.flags,
        recommended_action=state.recommended_action,
        explanation=state.explanation,
        steps_taken=state.steps_taken,
        audit_log=state.audit_log,
    )
