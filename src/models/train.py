"""
Model Training Module
=====================
Trains a churn prediction model (XGBoost or Logistic Regression) and
logs all parameters, metrics, and artefacts to MLflow.

MLflow features used
--------------------
- **Tracking**: parameters, metrics, confusion matrix artifact.
- **Model registry**: model registered under a versioned name.
- **MLflow Projects**: called via the MLproject entry point.

Usage
-----
    # Train XGBoost (default)
    python src/models/train.py

    # Train Logistic Regression baseline
    python src/models/train.py --model_type logistic_regression
"""

import argparse
import logging
import os

import mlflow
import mlflow.sklearn
import mlflow.xgboost
import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path: str = "conf/config.yaml") -> dict:
    """Load project configuration.

    Args:
        config_path: Path to YAML config file.

    Returns:
        Configuration dictionary.
    """
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_split(split: str, target: str) -> tuple[pd.DataFrame, pd.Series]:
    """Load a processed feature split and return X, y.

    Args:
        split: One of 'train', 'val', 'test'.
        target: Name of the target column.

    Returns:
        Tuple of (features DataFrame, target Series).

    Raises:
        FileNotFoundError: If the feature file does not exist.
    """
    path = f"data/processed/{split}_features.csv"
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Feature file not found: {path}. " "Run feature_engineering.py first."
        )
    df = pd.read_csv(path)
    X = df.drop(columns=[target])
    y = df[target]
    return X, y


def build_model(model_type: str, cfg: dict):
    """Instantiate a model based on model_type.

    Args:
        model_type: Either 'xgboost' or 'logistic_regression'.
        cfg: Full config dictionary.

    Returns:
        Unfitted sklearn-compatible model.

    Raises:
        ValueError: If model_type is not recognised.
    """
    if model_type == "xgboost":
        params = cfg["model"]["xgboost"]
        return XGBClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            learning_rate=params["learning_rate"],
            scale_pos_weight=params["scale_pos_weight"],
            random_state=params["random_state"],
            eval_metric="logloss",
            use_label_encoder=False,
        )
    elif model_type == "logistic_regression":
        params = cfg["model"]["logistic_regression"]
        return LogisticRegression(
            max_iter=params["max_iter"],
            class_weight=params["class_weight"],
            random_state=params["random_state"],
        )
    else:
        raise ValueError(
            f"Unknown model_type '{model_type}'. "
            "Choose 'xgboost' or 'logistic_regression'."
        )


def compute_metrics(y_true: pd.Series, y_pred: np.ndarray, y_proba: np.ndarray) -> dict:
    """Compute churn prediction evaluation metrics.

    Recall is the primary metric (missing a churner is costly).

    Args:
        y_true: Ground truth labels.
        y_pred: Binary predictions.
        y_proba: Predicted probabilities for the positive class.

    Returns:
        Dictionary of metric name → value.
    """
    return {
        "recall": recall_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_proba),
    }


def main(model_type: str = "xgboost") -> None:
    """Train, evaluate, and register the churn model with MLflow.

    Args:
        model_type: Model architecture to train.
    """
    cfg = load_config()
    target = cfg["features"]["target_column"]
    mlflow_cfg = cfg["mlflow"]

    mlflow.set_tracking_uri(mlflow_cfg["tracking_uri"])
    mlflow.set_experiment(mlflow_cfg["experiment_name"])

    X_train, y_train = load_split("train", target)
    X_val, y_val = load_split("val", target)
    X_test, y_test = load_split("test", target)

    logger.info("Training %s on %d samples...", model_type, len(X_train))

    with mlflow.start_run(run_name=model_type):
        # --- Log parameters ---
        model_params = cfg["model"].get(model_type, {})
        mlflow.log_params({"model_type": model_type, **model_params})

        # --- Train ---
        model = build_model(model_type, cfg)
        model.fit(X_train, y_train)

        # --- Evaluate on validation set ---
        val_preds = model.predict(X_val)
        val_proba = model.predict_proba(X_val)[:, 1]
        val_metrics = compute_metrics(y_val, val_preds, val_proba)
        mlflow.log_metrics({f"val_{k}": v for k, v in val_metrics.items()})
        logger.info("Validation metrics: %s", val_metrics)

        # --- Evaluate on test set ---
        test_preds = model.predict(X_test)
        test_proba = model.predict_proba(X_test)[:, 1]
        test_metrics = compute_metrics(y_test, test_preds, test_proba)
        mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})
        logger.info("Test metrics: %s", test_metrics)

        # --- Log classification report as artifact ---
        report = classification_report(
            y_test, test_preds, target_names=["No Churn", "Churn"]
        )
        os.makedirs("mlruns/artifacts", exist_ok=True)
        report_path = "mlruns/artifacts/classification_report.txt"
        with open(report_path, "w") as f:
            f.write(report)
        mlflow.log_artifact(report_path)

        # --- Log model to MLflow registry ---
        if model_type == "xgboost":
            mlflow.xgboost.log_model(
                model,
                artifact_path="model",
                registered_model_name=mlflow_cfg["model_name"],
            )
        else:
            mlflow.sklearn.log_model(
                model,
                artifact_path="model",
                registered_model_name=mlflow_cfg["model_name"],
            )

        logger.info(
            "Run complete. Model registered as '%s'.",
            mlflow_cfg["model_name"],
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train churn prediction model.")
    parser.add_argument(
        "--model_type",
        type=str,
        default="xgboost",
        choices=["xgboost", "logistic_regression"],
        help="Model architecture to train.",
    )
    args = parser.parse_args()
    main(model_type=args.model_type)
