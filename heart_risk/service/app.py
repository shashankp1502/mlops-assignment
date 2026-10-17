"""
FastAPI serving layer for the heart-risk classifier.

A few things here are deliberately different from a "minimal" FastAPI
setup:

- Logging is emitted as single-line JSON (via `JsonLogFormatter`) rather
  than a plain f-string, which is closer to what you'd actually want
  feeding into a log aggregator (Loki, CloudWatch, etc.) in production.
- `/model-info` exposes the MLflow Model Registry name/version alongside
  the holdout metrics, since the training pipeline registers the champion
  model rather than just pickling it.
- Prometheus metric names are namespaced `heart_risk_*` to avoid colliding
  with any other service's metrics on a shared Prometheus instance.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

from heart_risk.modeling.inference import DEFAULT_MODEL_DIR, RiskPredictor
from heart_risk.service.contracts import PatientRecord, RiskAssessment, ServiceStatus

METADATA_PATH = Path(DEFAULT_MODEL_DIR).parent / "metadata.json"


class JsonLogFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if hasattr(record, "extra_fields"):
            payload.update(record.extra_fields)
        return json.dumps(payload)


handler = logging.StreamHandler()
handler.setFormatter(JsonLogFormatter())
logger = logging.getLogger("heart_risk.service")
logger.setLevel(logging.INFO)
logger.addHandler(handler)
logger.propagate = False


def log_event(message: str, **fields) -> None:
    record = logging.LogRecord(logger.name, logging.INFO, "", 0, message, None, None)
    record.extra_fields = fields
    logger.handle(record)


REQUEST_COUNTER = Counter(
    "heart_risk_http_requests_total", "Total HTTP requests", ["route", "method", "status"]
)
PREDICTION_COUNTER = Counter(
    "heart_risk_predictions_total", "Predictions served, by risk label", ["risk_label"]
)
LATENCY_HISTOGRAM = Histogram(
    "heart_risk_request_duration_seconds", "Request duration in seconds", ["route"]
)

_state: Dict[str, Any] = {"predictor": None, "metadata": {}}


def _load_metadata() -> dict:
    if METADATA_PATH.exists():
        return json.loads(METADATA_PATH.read_text())
    return {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    _state["metadata"] = _load_metadata()
    try:
        _state["predictor"] = RiskPredictor(DEFAULT_MODEL_DIR)
        log_event("model_loaded", model_dir=str(DEFAULT_MODEL_DIR))
    except FileNotFoundError as exc:
        log_event("model_load_failed", error=str(exc))
        _state["predictor"] = None
    yield
    _state.clear()


app = FastAPI(
    title="Heart Risk Prediction Service",
    description="Predicts cardiovascular disease risk from patient vitals (UCI Cleveland cohort).",
    version="0.1.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def observe_requests(request: Request, call_next):
    trace_id = uuid.uuid4().hex[:10]
    started_at = time.perf_counter()

    response = await call_next(request)

    elapsed = time.perf_counter() - started_at
    route = request.url.path
    LATENCY_HISTOGRAM.labels(route=route).observe(elapsed)
    REQUEST_COUNTER.labels(route=route, method=request.method, status=response.status_code).inc()

    log_event(
        "request_handled",
        trace_id=trace_id,
        method=request.method,
        route=route,
        status=response.status_code,
        duration_ms=round(elapsed * 1000, 2),
    )
    response.headers["X-Trace-Id"] = trace_id
    return response


@app.get("/", tags=["meta"])
def index():
    return {
        "service": "heart-risk-prediction-service",
        "docs": "/docs",
        "health": "/health",
        "model_info": "/model-info",
        "predict": "/predict (POST)",
        "metrics": "/metrics",
    }


@app.get("/health", response_model=ServiceStatus, tags=["meta"])
def health():
    predictor = _state.get("predictor")
    metadata = _state.get("metadata", {})
    return ServiceStatus(
        status="ok" if predictor is not None else "degraded",
        model_ready=predictor is not None,
        model_name=metadata.get("registered_model_name"),
        model_version=str(metadata.get("registered_model_version", "")) or None,
    )


@app.get("/model-info", tags=["meta"])
def model_info():
    metadata = _state.get("metadata", {})
    if not metadata:
        raise HTTPException(status_code=503, detail="Model metadata not available.")
    return {
        "champion_model": metadata.get("champion_model"),
        "registered_model_name": metadata.get("registered_model_name"),
        "registered_model_version": metadata.get("registered_model_version"),
        "holdout_metrics": metadata.get("holdout_metrics"),
        "best_params": metadata.get("best_params"),
        "trained_at_utc": metadata.get("trained_at_utc"),
    }


@app.post("/predict", response_model=RiskAssessment, tags=["inference"])
def predict(record: PatientRecord):
    predictor: RiskPredictor | None = _state.get("predictor")
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model is not loaded yet. Check /health.")

    result = predictor.predict_one(record.model_dump())
    PREDICTION_COUNTER.labels(risk_label=result["risk_label"]).inc()

    metadata = _state.get("metadata", {})
    return RiskAssessment(
        prediction=result["prediction"],
        risk_label=result["risk_label"],
        confidence=round(result["confidence"], 4),
        probability_disease=round(result["probability_disease"], 4),
        model_name=metadata.get("registered_model_name", "unknown"),
        model_version=str(metadata.get("registered_model_version", "unknown")),
    )


@app.get("/metrics", tags=["meta"])
def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
