import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import mlflow.sklearn
from fastapi.testclient import TestClient
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from heart_risk.dataset import split_xy
from heart_risk.transform import build_preprocessing_pipeline
from tests.fixtures import build_synthetic_frame

VALID_PAYLOAD = {
    "age": 58, "sex": 1, "cp": 2, "trestbps": 140, "chol": 211, "fbs": 0,
    "restecg": 0, "thalach": 165, "exang": 0, "oldpeak": 0.0, "slope": 1,
    "ca": 0, "thal": 3,
}


class ServiceEndpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp_dir = TemporaryDirectory()
        model_path = Path(cls.tmp_dir.name) / "model"
        metadata_path = Path(cls.tmp_dir.name) / "metadata.json"

        X, y = split_xy(build_synthetic_frame())
        pipeline = Pipeline(steps=[
            ("features", build_preprocessing_pipeline()),
            ("clf", DecisionTreeClassifier(random_state=0)),
        ])
        pipeline.fit(X, y)
        mlflow.sklearn.save_model(pipeline, str(model_path))
        metadata_path.write_text(json.dumps({
            "registered_model_name": "heart-risk-classifier",
            "registered_model_version": 1,
        }))

        import heart_risk.service.app as app_module
        cls._patches = [
            patch.object(app_module, "DEFAULT_MODEL_DIR", model_path),
            patch.object(app_module, "METADATA_PATH", metadata_path),
        ]
        for p in cls._patches:
            p.start()

        cls.client = TestClient(app_module.app)
        cls.client.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.client.__exit__(None, None, None)
        for p in cls._patches:
            p.stop()
        cls.tmp_dir.cleanup()

    def test_index_lists_routes(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("predict", response.json())

    def test_health_reports_model_ready(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["status"], "ok")
        self.assertTrue(body["model_ready"])

    def test_predict_valid_payload(self):
        response = self.client.post("/predict", json=VALID_PAYLOAD)
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertIn(body["prediction"], (0, 1))
        self.assertIn(body["risk_label"], ("elevated_risk", "low_risk"))
        self.assertTrue(0.0 <= body["confidence"] <= 1.0)

    def test_predict_missing_field_returns_422(self):
        bad = {k: v for k, v in VALID_PAYLOAD.items() if k != "age"}
        response = self.client.post("/predict", json=bad)
        self.assertEqual(response.status_code, 422)

    def test_predict_invalid_thal_returns_422(self):
        bad = {**VALID_PAYLOAD, "thal": 42}
        response = self.client.post("/predict", json=bad)
        self.assertEqual(response.status_code, 422)

    def test_trace_id_header_present(self):
        response = self.client.post("/predict", json=VALID_PAYLOAD)
        self.assertIn("X-Trace-Id", response.headers)

    def test_metrics_endpoint_exposes_namespaced_metrics(self):
        self.client.post("/predict", json=VALID_PAYLOAD)
        response = self.client.get("/metrics")
        self.assertEqual(response.status_code, 200)
        self.assertIn("heart_risk_http_requests_total", response.text)
        self.assertIn("heart_risk_predictions_total", response.text)


if __name__ == "__main__":
    unittest.main()
