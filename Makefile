.PHONY: install lint test ingest train serve docker-build docker-run helm-lint clean

PYTHON ?= python

install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

lint:
	flake8 heart_risk tests

test:
	$(PYTHON) -m unittest discover -s tests -p "test_*.py" -v

ingest:
	$(PYTHON) -m heart_risk.ingestion

train:
	$(PYTHON) -m heart_risk.modeling.train

serve:
	uvicorn heart_risk.service.app:app --reload --port 8080

docker-build:
	docker build -t heart-risk-api:latest -f deploy/docker/Dockerfile .

docker-run:
	docker run -d --name heart-risk-api -p 8080:8080 heart-risk-api:latest

helm-lint:
	helm lint deploy/helm/heart-risk-api

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache htmlcov .coverage
