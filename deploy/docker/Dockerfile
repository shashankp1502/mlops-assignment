# syntax=docker/dockerfile:1
#
# Build from the repo root:
#   docker build -t heart-risk-api:latest -f deploy/docker/Dockerfile .
#
# Needs artifacts/model/ (an MLflow-format model directory) and
# artifacts/metadata.json to already exist -- run `make train` first.

# ---- builder: install deps into an isolated venv ----------------------
FROM python:3.12-slim AS builder

WORKDIR /build
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY deploy/docker/requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ---- runtime: slim image, no compiler toolchain ------------------------
FROM python:3.12-slim

WORKDIR /srv
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY heart_risk/ heart_risk/
COPY artifacts/model/ artifacts/model/
COPY artifacts/metadata.json artifacts/metadata.json

RUN useradd --create-home runner && chown -R runner:runner /srv
USER runner

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

EXPOSE 8080

HEALTHCHECK --interval=15s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health').read()" || exit 1

# gunicorn managing uvicorn workers -- a more production-typical process
# model than invoking uvicorn directly, and gives us worker-count control
CMD ["gunicorn", "heart_risk.service.app:app", \
     "--workers", "2", \
     "--worker-class", "uvicorn.workers.UvicornWorker", \
     "--bind", "0.0.0.0:8080", \
     "--access-logfile", "-"]
