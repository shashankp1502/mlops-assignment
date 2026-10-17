"""Synthetic dataset builder shared across the unittest test cases.

Deliberately avoids the network and the real data/raw/heart_disease.csv
file -- the suite should pass on a clean checkout before `make ingest`
or `make train` have ever been run.
"""

import numpy as np
import pandas as pd

from heart_risk.transform import ALL_FEATURES

ROW_COUNT = 40


def build_synthetic_frame(seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    frame = pd.DataFrame({
        "age": rng.integers(29, 77, ROW_COUNT).astype(float),
        "sex": rng.integers(0, 2, ROW_COUNT).astype(float),
        "cp": rng.integers(1, 5, ROW_COUNT).astype(float),
        "trestbps": rng.integers(94, 200, ROW_COUNT).astype(float),
        "chol": rng.integers(126, 564, ROW_COUNT).astype(float),
        "fbs": rng.integers(0, 2, ROW_COUNT).astype(float),
        "restecg": rng.integers(0, 3, ROW_COUNT).astype(float),
        "thalach": rng.integers(71, 202, ROW_COUNT).astype(float),
        "exang": rng.integers(0, 2, ROW_COUNT).astype(float),
        "oldpeak": np.round(rng.uniform(0, 6.2, ROW_COUNT), 1),
        "slope": rng.integers(1, 4, ROW_COUNT).astype(float),
        "ca": rng.integers(0, 4, ROW_COUNT).astype(float),
        "thal": rng.choice([3.0, 6.0, 7.0], ROW_COUNT),
        "target": rng.integers(0, 2, ROW_COUNT),
    })

    # match the handful of real missing cells (ca/thal) seen in the
    # genuine UCI file
    frame.loc[3, "ca"] = np.nan
    frame.loc[9, "thal"] = np.nan

    assert set(ALL_FEATURES).issubset(frame.columns)
    return frame
