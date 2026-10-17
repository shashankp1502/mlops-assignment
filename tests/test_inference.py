import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import mlflow.sklearn
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from heart_risk.dataset import split_xy
from heart_risk.modeling.inference import RiskPredictor
from heart_risk.transform import build_preprocessing_pipeline
from tests.fixtures import build_synthetic_frame

VALID_RECORD = {
    "age": 54, "sex": 1, "cp": 3, "trestbps": 130, "chol": 246, "fbs": 0,
    "restecg": 0, "thalach": 150, "exang": 0, "oldpeak": 1.2, "slope": 2,
    "ca": 0, "thal": 3,
}


class RiskPredictorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp_dir = TemporaryDirectory()
        model_path = Path(cls.tmp_dir.name) / "model"

        X, y = split_xy(build_synthetic_frame())
        pipeline = Pipeline(steps=[
            ("features", build_preprocessing_pipeline()),
            ("clf", DecisionTreeClassifier(random_state=0)),
        ])
        pipeline.fit(X, y)
        mlflow.sklearn.save_model(pipeline, str(model_path))
        cls.model_path = model_path

    @classmethod
    def tearDownClass(cls):
        cls.tmp_dir.cleanup()

    def test_missing_model_directory_raises(self):
        with TemporaryDirectory() as empty_dir:
            with self.assertRaises(FileNotFoundError):
                RiskPredictor(Path(empty_dir) / "does-not-exist")

    def test_predict_one_returns_expected_keys(self):
        predictor = RiskPredictor(self.model_path)
        result = predictor.predict_one(VALID_RECORD)
        self.assertSetEqual(
            set(result.keys()),
            {"prediction", "risk_label", "confidence", "probability_disease"},
        )
        self.assertIn(result["prediction"], (0, 1))
        self.assertIn(result["risk_label"], ("elevated_risk", "low_risk"))

    def test_risk_label_matches_prediction(self):
        predictor = RiskPredictor(self.model_path)
        result = predictor.predict_one(VALID_RECORD)
        expected = "elevated_risk" if result["prediction"] == 1 else "low_risk"
        self.assertEqual(result["risk_label"], expected)

    def test_missing_field_raises(self):
        predictor = RiskPredictor(self.model_path)
        incomplete = {k: v for k, v in VALID_RECORD.items() if k != "chol"}
        with self.assertRaises(ValueError):
            predictor.predict_one(incomplete)

    def test_predict_many_returns_matching_length(self):
        predictor = RiskPredictor(self.model_path)
        results = predictor.predict_many([VALID_RECORD, VALID_RECORD, VALID_RECORD])
        self.assertEqual(len(results), 3)


if __name__ == "__main__":
    unittest.main()
