"""
FastAPI serving app for P7 predictive maintenance.

The model artifact is loaded ONCE at startup (via a lifespan handler) and reused
for every request. Endpoints:
  GET  /health          - is the service alive?
  GET  /ready           - is the model loaded and ready?
  POST /predict_failure - score a window of sensor readings for one asset.

No training-serving skew: features use the SAME compute_features as training,
and the z-score uses the baseline SAVED IN THE BUNDLE (not recomputed).
"""

from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.features.engineering import compute_baseline_zscore, compute_features

MODEL_PATH = Path("models") / "anomaly_model.joblib"

# Holds the loaded model bundle. Populated at startup; None until then.
_bundle: dict | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model artifact ONCE when the server starts (modern lifespan API)."""
    global _bundle
    if MODEL_PATH.exists():
        _bundle = joblib.load(MODEL_PATH)
    yield
    # (nothing to clean up on shutdown for now)


app = FastAPI(title="P7 Predictive Maintenance API", version="0.1.0", lifespan=lifespan)


class PredictRequest(BaseModel):
    """A window of recent readings for ONE asset (oldest first)."""

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
    """Transparent, auditable mapping from risk score to a next step."""
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


@app.post("/predict_failure", response_model=PredictResponse)
def predict_failure(req: PredictRequest) -> PredictResponse:
    if _bundle is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

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
