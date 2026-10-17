"""
Trains and compares two tree-based classifiers for heart disease risk --
a single Decision Tree and a Gradient Boosting ensemble -- with
randomized hyperparameter search, logging every run to an MLflow
tracking server backed by SQLite (rather than the default local file
store), which also lets us register the winning model in the MLflow
Model Registry.

Usage:
    python -m heart_risk.modeling.train
    make train
"""

from __future__ import annotations

import json
import logging
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

import mlflow
import mlflow.sklearn
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from heart_risk.dataset import load_dataset, split_xy, stratified_holdout
from heart_risk.modeling.metrics import (
    compute_metrics,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_roc_curve,
)
from heart_risk.transform import build_preprocessing_pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
MODEL_DIR = ARTIFACT_DIR / "model"
METADATA_PATH = ARTIFACT_DIR / "metadata.json"

MLFLOW_TRACKING_URI = f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}"
EXPERIMENT_NAME = "heart-risk-classification"
REGISTERED_MODEL_NAME = "heart-risk-classifier"

SEED = 7
CV_FOLDS = 10
SEARCH_ITERATIONS = 20
PRIMARY_METRIC = "f1"


@dataclass
class CandidateSpec:
    name: str
    pipeline: Pipeline
    param_distributions: dict


def build_candidates() -> Dict[str, CandidateSpec]:
    tree_pipeline = Pipeline(steps=[
        ("features", build_preprocessing_pipeline()),
        ("clf", DecisionTreeClassifier(random_state=SEED)),
    ])
    tree_params = {
        "clf__max_depth": [3, 4, 5, 6, 8, None],
        "clf__min_samples_split": [2, 5, 10, 20],
        "clf__min_samples_leaf": [1, 2, 4, 8],
        "clf__criterion": ["gini", "entropy"],
    }

    boosting_pipeline = Pipeline(steps=[
        ("features", build_preprocessing_pipeline()),
        ("clf", GradientBoostingClassifier(random_state=SEED)),
    ])
    boosting_params = {
        "clf__n_estimators": [50, 100, 150, 200],
        "clf__learning_rate": [0.01, 0.05, 0.1, 0.2],
        "clf__max_depth": [2, 3, 4],
        "clf__subsample": [0.7, 0.85, 1.0],
    }

    return {
        "decision_tree": CandidateSpec("decision_tree", tree_pipeline, tree_params),
        "gradient_boosting": CandidateSpec("gradient_boosting", boosting_pipeline, boosting_params),
    }


def _get_feature_names(fitted_pipeline: Pipeline):
    return fitted_pipeline.named_steps["features"].named_steps["columns"].get_feature_names_out()


def run_candidate(spec: CandidateSpec, X_train, y_train, X_holdout, y_holdout) -> dict:
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)

    with mlflow.start_run(run_name=spec.name) as run:
        mlflow.set_tag("algorithm_family", spec.name)
        mlflow.log_params({"cv_folds": CV_FOLDS, "search_iterations": SEARCH_ITERATIONS})

        search = RandomizedSearchCV(
            spec.pipeline,
            param_distributions=spec.param_distributions,
            n_iter=SEARCH_ITERATIONS,
            scoring=PRIMARY_METRIC,
            cv=cv,
            random_state=SEED,
            n_jobs=4,  # fixed (not -1) to avoid noisy joblib/loky resource-tracker warnings on macOS
            refit=True,
        )
        logger.info("[%s] searching %d candidates over %d-fold CV...", spec.name, SEARCH_ITERATIONS, CV_FOLDS)
        search.fit(X_train, y_train)

        best_model = search.best_estimator_
        mlflow.log_params({f"best__{k}": v for k, v in search.best_params_.items()})
        mlflow.log_metric(f"cv_{PRIMARY_METRIC}_mean", search.best_score_)

        y_pred = best_model.predict(X_holdout)
        y_proba = best_model.predict_proba(X_holdout)[:, 1]
        holdout_metrics = compute_metrics(y_holdout, y_pred, y_proba)
        mlflow.log_metrics({f"holdout_{k}": v for k, v in holdout_metrics.items()})

        with tempfile.TemporaryDirectory() as tmp:
            cm_path = plot_confusion_matrix(
                y_holdout, y_pred, Path(tmp) / "confusion_matrix.png", f"{spec.name} — confusion matrix"
            )
            roc_path = plot_roc_curve(
                y_holdout, y_proba, Path(tmp) / "roc_curve.png", f"{spec.name} — ROC curve"
            )
            mlflow.log_artifact(str(cm_path))
            mlflow.log_artifact(str(roc_path))

            classifier = best_model.named_steps["clf"]
            if hasattr(classifier, "feature_importances_"):
                fi_path = plot_feature_importance(
                    _get_feature_names(best_model),
                    classifier.feature_importances_,
                    Path(tmp) / "feature_importance.png",
                    f"{spec.name} — feature importance",
                )
                mlflow.log_artifact(str(fi_path))

        mlflow.sklearn.log_model(best_model, artifact_path="model", input_example=X_train.head(2))

        logger.info(
            "[%s] best_params=%s | cv_%s=%.4f | holdout_f1=%.4f | holdout_roc_auc=%.4f",
            spec.name, search.best_params_, PRIMARY_METRIC, search.best_score_,
            holdout_metrics["f1"], holdout_metrics["roc_auc"],
        )

        return {
            "name": spec.name,
            "estimator": best_model,
            "best_params": search.best_params_,
            f"cv_{PRIMARY_METRIC}_mean": search.best_score_,
            "holdout_metrics": holdout_metrics,
            "run_id": run.info.run_id,
        }


def main() -> int:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    frame = load_dataset()
    features, target = split_xy(frame)
    X_train, X_holdout, y_train, y_holdout = stratified_holdout(features, target)
    logger.info("Train/holdout split: %d / %d rows", len(X_train), len(X_holdout))

    results = [
        run_candidate(spec, X_train, y_train, X_holdout, y_holdout)
        for spec in build_candidates().values()
    ]

    champion = max(results, key=lambda r: r["holdout_metrics"][PRIMARY_METRIC])
    champion_score = champion["holdout_metrics"][PRIMARY_METRIC]
    logger.info("Champion: %s (holdout_%s=%.4f)", champion["name"], PRIMARY_METRIC, champion_score)

    model_uri = f"runs:/{champion['run_id']}/model"
    with mlflow.start_run(run_id=champion["run_id"]):
        mlflow.set_tag("champion", "true")
    registered = mlflow.register_model(model_uri, REGISTERED_MODEL_NAME)
    logger.info("Registered champion as '%s' version %s", REGISTERED_MODEL_NAME, registered.version)

    if MODEL_DIR.exists():
        import shutil
        shutil.rmtree(MODEL_DIR)
    mlflow.sklearn.save_model(champion["estimator"], str(MODEL_DIR))

    metadata = {
        "champion_model": champion["name"],
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "features": list(X_train.columns),
        "best_params": champion["best_params"],
        f"cv_{PRIMARY_METRIC}_mean": champion[f"cv_{PRIMARY_METRIC}_mean"],
        "holdout_metrics": champion["holdout_metrics"],
        "mlflow_run_id": champion["run_id"],
        "registered_model_name": REGISTERED_MODEL_NAME,
        "registered_model_version": registered.version,
        "all_candidates": [
            {
                "name": r["name"],
                "holdout_metrics": r["holdout_metrics"],
                f"cv_{PRIMARY_METRIC}_mean": r[f"cv_{PRIMARY_METRIC}_mean"],
            }
            for r in results
        ],
    }
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_PATH.write_text(json.dumps(metadata, indent=2))

    logger.info("Saved MLflow-format model -> %s", MODEL_DIR)
    logger.info("Saved metadata -> %s", METADATA_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
