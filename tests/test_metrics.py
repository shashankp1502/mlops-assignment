import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from heart_risk.modeling.metrics import (
    compute_metrics,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_roc_curve,
)


class ComputeMetricsTests(unittest.TestCase):
    def test_perfect_predictions(self):
        y_true = [0, 1, 0, 1, 1]
        y_pred = [0, 1, 0, 1, 1]
        y_proba = [0.05, 0.9, 0.1, 0.8, 0.95]
        metrics = compute_metrics(y_true, y_pred, y_proba)
        for key in ("accuracy", "precision", "recall", "f1", "roc_auc"):
            self.assertAlmostEqual(metrics[key], 1.0)

    def test_metrics_within_valid_range(self):
        rng = np.random.default_rng(3)
        y_true = rng.integers(0, 2, 60)
        y_pred = rng.integers(0, 2, 60)
        y_proba = rng.uniform(0, 1, 60)
        metrics = compute_metrics(y_true, y_pred, y_proba)
        for key, value in metrics.items():
            self.assertGreaterEqual(value, 0.0, key)
            self.assertLessEqual(value, 1.0, key)

    def test_single_class_predictions_do_not_crash(self):
        y_true = [0, 0, 1, 1]
        y_pred = [0, 0, 0, 0]
        y_proba = [0.1, 0.2, 0.3, 0.4]
        metrics = compute_metrics(y_true, y_pred, y_proba)
        self.assertEqual(metrics["precision"], 0.0)
        self.assertEqual(metrics["recall"], 0.0)


class PlottingTests(unittest.TestCase):
    def test_confusion_matrix_file_created(self):
        with TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "cm.png"
            result = plot_confusion_matrix([0, 1, 0, 1], [0, 1, 1, 1], out_path, "test")
            self.assertTrue(result.exists())
            self.assertGreater(result.stat().st_size, 0)

    def test_roc_curve_file_created(self):
        with TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "roc.png"
            result = plot_roc_curve([0, 1, 0, 1], [0.2, 0.9, 0.3, 0.7], out_path, "test")
            self.assertTrue(result.exists())
            self.assertGreater(result.stat().st_size, 0)

    def test_feature_importance_file_created(self):
        with TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "fi.png"
            result = plot_feature_importance(
                ["age", "chol", "thal"], [0.5, 0.3, 0.2], out_path, "test"
            )
            self.assertTrue(result.exists())
            self.assertGreater(result.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
