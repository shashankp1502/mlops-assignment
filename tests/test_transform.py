import unittest

import numpy as np

from heart_risk.dataset import split_xy
from heart_risk.transform import (
    ALL_FEATURES,
    BINARY_FEATURES,
    NUMERIC_FEATURES,
    ORDINAL_FEATURES,
    AgeGroupBucketer,
    build_preprocessing_pipeline,
)
from tests.fixtures import build_synthetic_frame


class FeatureGroupCoverageTests(unittest.TestCase):
    def test_groups_cover_all_features_exactly_once(self):
        grouped = NUMERIC_FEATURES + ORDINAL_FEATURES + BINARY_FEATURES
        self.assertListEqual(sorted(grouped), sorted(ALL_FEATURES))
        self.assertEqual(len(grouped), len(set(grouped)))


class AgeGroupBucketerTests(unittest.TestCase):
    def test_buckets_assigned_correctly(self):
        X, _ = split_xy(build_synthetic_frame())
        bucketer = AgeGroupBucketer()
        transformed = bucketer.fit_transform(X)

        for age, group in zip(transformed["age"], transformed["age_group"]):
            if age < 45:
                self.assertEqual(group, 0)
            elif age < 60:
                self.assertEqual(group, 1)
            else:
                self.assertEqual(group, 2)

    def test_adds_exactly_one_column(self):
        X, _ = split_xy(build_synthetic_frame())
        bucketer = AgeGroupBucketer()
        transformed = bucketer.fit_transform(X)
        self.assertEqual(transformed.shape[1], X.shape[1] + 1)


class PreprocessingPipelineTests(unittest.TestCase):
    def setUp(self):
        self.X, _ = split_xy(build_synthetic_frame())

    def test_no_nan_in_output_despite_missing_input(self):
        self.assertTrue(self.X.isna().any().any(), "fixture should contain NaNs for this test to matter")
        pipeline = build_preprocessing_pipeline()
        transformed = pipeline.fit_transform(self.X)
        self.assertFalse(np.isnan(transformed).any())

    def test_output_uses_ordinal_not_onehot_encoding(self):
        """Ordinal encoding should NOT expand column count the way
        one-hot would -- output width should be input width + 1 (for
        the derived age_group column), not dozens of dummy columns."""
        pipeline = build_preprocessing_pipeline()
        transformed = pipeline.fit_transform(self.X)
        self.assertEqual(transformed.shape[1], len(ALL_FEATURES) + 1)

    def test_row_count_preserved(self):
        pipeline = build_preprocessing_pipeline()
        transformed = pipeline.fit_transform(self.X)
        self.assertEqual(transformed.shape[0], self.X.shape[0])

    def test_independent_refits_produce_consistent_width(self):
        half = len(self.X) // 2
        pipe_a = build_preprocessing_pipeline()
        pipe_b = build_preprocessing_pipeline()
        out_a = pipe_a.fit_transform(self.X.iloc[:half])
        out_b = pipe_b.fit_transform(self.X.iloc[half:])
        self.assertEqual(out_a.shape[1], out_b.shape[1])


if __name__ == "__main__":
    unittest.main()
