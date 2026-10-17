"""Structural tests for heart_risk.modeling.train -- doesn't run the full
RandomizedSearchCV (slow; exercised as its own CI stage via `make train`
against the real dataset). Just checks the candidate pipelines are wired
correctly and can fit/predict once on small data."""

import unittest

from sklearn.pipeline import Pipeline

from heart_risk.dataset import split_xy
from heart_risk.modeling.train import build_candidates
from tests.fixtures import build_synthetic_frame


class BuildCandidatesTests(unittest.TestCase):
    def test_both_expected_models_present(self):
        candidates = build_candidates()
        self.assertSetEqual(set(candidates.keys()), {"decision_tree", "gradient_boosting"})

    def test_each_candidate_is_a_two_step_pipeline(self):
        for name, spec in build_candidates().items():
            self.assertIsInstance(spec.pipeline, Pipeline, name)
            self.assertListEqual([s[0] for s in spec.pipeline.steps], ["features", "clf"])

    def test_param_distributions_use_clf_prefix(self):
        for name, spec in build_candidates().items():
            self.assertTrue(spec.param_distributions, f"{name} has empty param grid")
            for key in spec.param_distributions:
                self.assertTrue(key.startswith("clf__"), f"{name}: {key}")


class CandidateFitPredictTests(unittest.TestCase):
    def setUp(self):
        self.X, self.y = split_xy(build_synthetic_frame())

    def test_decision_tree_fits_and_predicts(self):
        spec = build_candidates()["decision_tree"]
        spec.pipeline.fit(self.X, self.y)
        preds = spec.pipeline.predict(self.X)
        self.assertEqual(len(preds), len(self.X))
        self.assertTrue(set(preds).issubset({0, 1}))

    def test_gradient_boosting_fits_and_predicts(self):
        spec = build_candidates()["gradient_boosting"]
        spec.pipeline.fit(self.X, self.y)
        preds = spec.pipeline.predict(self.X)
        self.assertEqual(len(preds), len(self.X))
        self.assertTrue(set(preds).issubset({0, 1}))


if __name__ == "__main__":
    unittest.main()
