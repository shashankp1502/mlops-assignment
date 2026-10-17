"""Loading + splitting helpers shared by training and tests."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from heart_risk.transform import ALL_FEATURES

TARGET = "target"
DEFAULT_DATA_PATH = Path("data/raw/heart_disease.csv")


def load_dataset(path: Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"No dataset at {path}. Run `make ingest` (or "
            "`python -m heart_risk.ingestion`) first."
        )
    return pd.read_csv(path)


def split_xy(frame: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    missing = set(ALL_FEATURES + [TARGET]) - set(frame.columns)
    if missing:
        raise ValueError(f"Dataset is missing expected column(s): {sorted(missing)}")
    return frame[ALL_FEATURES].copy(), frame[TARGET].copy()


def stratified_holdout(
    features: pd.DataFrame,
    target: pd.Series,
    holdout_fraction: float = 0.2,
    seed: int = 7,
):
    """Stratified train/holdout split. Named 'holdout' rather than 'test'
    throughout this module to keep it unambiguous from the k-fold CV test
    folds used during RandomizedSearchCV."""
    return train_test_split(
        features, target, test_size=holdout_fraction, random_state=seed, stratify=target
    )
