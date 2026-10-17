import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from heart_risk.dataset import TARGET, load_dataset, split_xy, stratified_holdout
from heart_risk.transform import ALL_FEATURES
from tests.fixtures import build_synthetic_frame


class LoadDatasetTests(unittest.TestCase):
    def test_missing_file_raises(self):
        with TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                load_dataset(Path(tmp) / "nope.csv")

    def test_loads_existing_file(self):
        frame = build_synthetic_frame()
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "heart.csv"
            frame.to_csv(path, index=False)
            loaded = load_dataset(path)
            self.assertEqual(loaded.shape, frame.shape)


class SplitXYTests(unittest.TestCase):
    def setUp(self):
        self.frame = build_synthetic_frame()

    def test_columns_match_feature_list(self):
        X, y = split_xy(self.frame)
        self.assertListEqual(list(X.columns), ALL_FEATURES)
        self.assertEqual(y.name, TARGET)

    def test_raises_on_missing_column(self):
        broken = self.frame.drop(columns=["chol"])
        with self.assertRaises(ValueError):
            split_xy(broken)


class StratifiedHoldoutTests(unittest.TestCase):
    def setUp(self):
        self.frame = build_synthetic_frame()
        self.X, self.y = split_xy(self.frame)

    def test_split_sizes(self):
        X_train, X_hold, y_train, y_hold = stratified_holdout(self.X, self.y, holdout_fraction=0.25)
        self.assertEqual(len(X_train) + len(X_hold), len(self.X))
        self.assertEqual(len(X_hold), round(0.25 * len(self.X)))

    def test_stratification_preserves_balance(self):
        X_train, X_hold, y_train, y_hold = stratified_holdout(self.X, self.y, holdout_fraction=0.5)
        full_ratio = self.y.mean()
        self.assertLessEqual(abs(y_train.mean() - full_ratio), 0.15)
        self.assertLessEqual(abs(y_hold.mean() - full_ratio), 0.15)


if __name__ == "__main__":
    unittest.main()
