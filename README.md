# predictive-maintenance-triage

Production-grade predictive maintenance system for manufacturing / IoT.
Detects impending equipment failure from multivariate sensor data and runs a
governed, human-in-the-loop maintenance workflow via a stateful AI agent.

## What this system does
- **Ingests** sensor telemetry in micro-batches with validated data contracts
- **Scores** every asset for failure risk using anomaly-detection models
- **Serves** predictions via a FastAPI endpoint (`/predict_failure`)
- **Monitors** for sensor drift and silent feed failures (Prometheus + Grafana + Evidently)
- **Explains** risk with a grounded, LLM-generated maintenance report
- **Acts** through a LangGraph Maintenance Scheduler Agent with human sign-off on critical assets

## Engineering stack
Python 3.12 - FastAPI - scikit-learn / LightGBM - LangGraph - MLflow - DVC
Docker - GitHub Actions CI - Prometheus / Grafana - Evidently AI - Langfuse - Power BI

## Engineering lifecycle (7 stages)
Template -> Ingestion & Contracts -> Evaluation-as-Tests -> Serving ->
Monitoring -> GenAI Integration -> Agentic Layer


---
