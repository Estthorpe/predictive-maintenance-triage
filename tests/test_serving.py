"""
Tests for the serving API, using FastAPI''s TestClient (no live server needed).

These run the app in-process, trigger the startup hook (which loads the model
artifact), and call the endpoints directly - so CI can verify serving works.
"""

import pytest
from fastapi.testclient import TestClient

from src.serving.app import app


@pytest.fixture(scope="module")
def client():
    # 'with' triggers FastAPI startup/shutdown events -> model loads.
    with TestClient(app) as c:
        yield c


def test_health_is_alive(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "alive"


def test_ready_reports_model_loaded(client):
    r = client.get("/ready")
    assert r.status_code == 200
    assert r.json()["status"] == "ready"


def test_predict_returns_a_risk_score(client):
    payload = {"device_id": "BEARING_07", "values": [10.1, 10.3, 9.8, 11.2, 14.5, 18.9]}
    r = client.post("/predict_failure", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert body["device_id"] == "BEARING_07"
    assert 0.0 <= body["risk_score"] <= 1.0
    assert len(body["top_sensors"]) > 0
    assert isinstance(body["recommended_action"], str)


def test_predict_rejects_empty_values(client):
    # Empty values list violates the contract (min_length=1) -> 422.
    r = client.post("/predict_failure", json={"device_id": "X", "values": []})
    assert r.status_code == 422


def test_predict_rejects_missing_device_id(client):
    # Missing required field -> 422 validation error.
    r = client.post("/predict_failure", json={"values": [10.0, 11.0]})
    assert r.status_code == 422
