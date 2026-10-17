"""
Loads the packaged champion model (an MLflow-format model directory, not
a raw pickle) and serves predictions for new patient records.

Using `mlflow.pyfunc.load_model` rather than `joblib.load` means the
loading code doesn't need to know anything about sklearn internals, and
the exact same loader would work unchanged if a future run swapped in an
XGBoost or ONNX-flavored model logged through MLflow.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import mlflow.pyfunc
import pandas as pd

from heart_risk.transform import ALL_FEATURES

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_DIR = PROJECT_ROOT / "artifacts" / "model"


class RiskPredictor:
    """Wraps the loaded MLflow pyfunc model with patient-record-shaped
    convenience methods, so the API layer never touches a DataFrame."""

    def __init__(self, model_dir: str | Path = DEFAULT_MODEL_DIR):
        self.model_dir = Path(model_dir)
        if not self.model_dir.exists():
            raise FileNotFoundError(
                f"No model directory at {self.model_dir}. Run `make train` "
                "(or `python -m heart_risk.modeling.train`) first."
            )
        self._model = mlflow.pyfunc.load_model(str(self.model_dir))
        # pyfunc's generic .predict() only returns hard class labels, so
        # for predict_proba we reach into the wrapped sklearn pipeline
        # directly (see predict_one below)
        self._sklearn_model = self._model._model_impl.sklearn_model

    def _to_frame(self, record: Dict) -> pd.DataFrame:
        missing = set(ALL_FEATURES) - set(record.keys())
        if missing:
            raise ValueError(f"Missing required field(s): {sorted(missing)}")
        return pd.DataFrame([{col: record[col] for col in ALL_FEATURES}])

    def predict_one(self, record: Dict) -> Dict:
        row = self._to_frame(record)
        probability_disease = float(self._sklearn_model.predict_proba(row)[0, 1])
        prediction = int(probability_disease >= 0.5)

        return {
            "prediction": prediction,
            "risk_label": "elevated_risk" if prediction == 1 else "low_risk",
            "confidence": probability_disease if prediction == 1 else 1 - probability_disease,
            "probability_disease": probability_disease,
        }

    def predict_many(self, records: List[Dict]) -> List[Dict]:
        return [self.predict_one(r) for r in records]
