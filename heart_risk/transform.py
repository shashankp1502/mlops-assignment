"""
Feature engineering for the heart_risk project.

Design choices (documented here since they differ from a "default"
pipeline, and the reasoning matters for the report):

- **KNNImputer** (k=5) for the continuous columns instead of simple
  median imputation -- with only 6 missing cells total in the whole
  dataset, borrowing values from the 5 nearest patients (by the other
  numeric features) is a reasonable step up from "just use the median"
  and costs nothing at this data size.
- **Ordinal encoding** (not one-hot) for the nominal categoricals
  (cp, restecg, slope, thal). This is deliberate given the two models
  trained downstream (Decision Tree, Gradient Boosting) are both
  tree-based: trees split on thresholds, so an integer-coded category
  doesn't impose a false distance metric the way it would for a linear
  model -- it just gives the tree one more axis to split on. Skipping
  one-hot also keeps the feature space small on an already-tiny dataset
  (303 rows), which helps avoid overfitting to sparse one-hot columns.
- **Derived `age_group` feature**: bucket age into
  {young: <45, middle: 45-59, senior: 60+} -- a simple, clinically
  motivated engineered feature on top of raw age, added as an extra
  ordinal column.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]
ORDINAL_FEATURES = ["cp", "restecg", "slope", "thal", "ca"]
BINARY_FEATURES = ["sex", "fbs", "exang"]

ALL_FEATURES = NUMERIC_FEATURES + ORDINAL_FEATURES + BINARY_FEATURES

AGE_BINS = [0, 44, 59, 150]
AGE_LABELS = [0, 1, 2]  # 0=young(<45), 1=middle(45-59), 2=senior(60+)


@dataclass
class AgeGroupBucketer(BaseEstimator, TransformerMixin):
    """Adds a single derived 'age_group' column computed from 'age'.

    Kept as its own small transformer (rather than inline pandas code in
    a notebook) so it's part of the serialized sklearn Pipeline and runs
    identically at training and inference time.
    """

    age_column: str = "age"

    def fit(self, X: pd.DataFrame, y=None):  # noqa: N803 (sklearn convention)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X["age_group"] = pd.cut(
            X[self.age_column], bins=AGE_BINS, labels=AGE_LABELS, right=True
        ).astype(float)
        return X

    def get_feature_names_out(self, input_features=None):
        base = list(input_features) if input_features is not None else []
        return np.array(base + ["age_group"])


def build_preprocessing_pipeline() -> Pipeline:
    """Returns a two-stage pipeline: add the engineered age_group column,
    then feed everything through a ColumnTransformer."""

    numeric_pipeline = Pipeline(steps=[
        ("impute", KNNImputer(n_neighbors=5)),
        ("scale", StandardScaler()),
    ])

    ordinal_pipeline = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
    ])

    binary_pipeline = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="most_frequent")),
    ])

    engineered_pipeline = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="most_frequent")),
    ])

    column_transformer = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("ordinal", ordinal_pipeline, ORDINAL_FEATURES),
            ("binary", binary_pipeline, BINARY_FEATURES),
            ("engineered", engineered_pipeline, ["age_group"]),
        ],
        remainder="drop",
    )

    return Pipeline(steps=[
        ("age_bucket", AgeGroupBucketer()),
        ("columns", column_transformer),
    ])
