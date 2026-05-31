"""
Inference Module
================
Loads a registered model from the MLflow Model Registry (or any
MLflow model URI) and scores new customer data.

Outputs a CSV with the original features plus two new columns:
- ``churn_prediction``: binary label (0 = no churn, 1 = churn)
- ``churn_probability``: probability of churn (float 0–1)

Usage
-----
    # Score using the latest production model
    python src/models/inference.py \\
        --model_uri "models:/churn-predictor/latest" \\
        --input_path data/processed/test_features.csv

    # Score using a specific run artifact
    python src/models/inference.py \\
        --model_uri "runs:/<run_id>/model" \\
        --input_path data/processed/test_features.csv
"""

import argparse
import logging
import os

import mlflow
import pandas as pd
import yaml

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


def load_model(model_uri: str):
    """Load a model from MLflow using a model URI.

    Args:
        model_uri: MLflow model URI, e.g. 'models:/churn-predictor/latest'
                   or 'runs:/<run_id>/model'.

    Returns:
        Loaded MLflow pyfunc model.
    """
    logger.info("Loading model from: %s", model_uri)
    model = mlflow.pyfunc.load_model(model_uri)
    logger.info("Model loaded successfully.")
    return model


def score(
    model, input_path: str, target_column: str, tracking_uri: str
) -> pd.DataFrame:
    """Score customer data and return predictions.

    The target column is dropped before scoring if present (supports
    passing labelled test data directly).

    Args:
        model: Loaded MLflow pyfunc model.
        input_path: Path to CSV file with customer features.
        target_column: Name of the target column to exclude from features.
        tracking_uri: MLflow tracking URI (for config only).

    Returns:
        DataFrame with original data plus prediction columns.
    """
    df = pd.read_csv(input_path)
    logger.info("Loaded %d rows from %s", len(df), input_path)

    X = df.drop(columns=[target_column], errors="ignore")

    predictions = model.predict(X)
    # pyfunc returns numpy array; get probabilities via wrapped model
    raw_model = model._model_impl  # unwrap to access predict_proba
    try:
        probabilities = raw_model.predict_proba(X)[:, 1]
    except AttributeError:
        logger.warning("predict_proba not available; using binary predictions only.")
        probabilities = predictions.astype(float)

    df["churn_prediction"] = predictions
    df["churn_probability"] = probabilities

    churn_count = int(df["churn_prediction"].sum())
    logger.info(
        "Scored %d customers | Predicted churners: %d (%.1f%%)",
        len(df),
        churn_count,
        100 * churn_count / len(df),
    )
    return df


def main(model_uri: str, input_path: str) -> None:
    """Main inference pipeline.

    Args:
        model_uri: MLflow model URI.
        input_path: Path to input CSV.
    """
    cfg = load_config()
    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])

    model = load_model(model_uri)
    results = score(
        model,
        input_path,
        target_column=cfg["features"]["target_column"],
        tracking_uri=cfg["mlflow"]["tracking_uri"],
    )

    output_path = input_path.replace(".csv", "_scored.csv")
    results.to_csv(output_path, index=False)
    logger.info("Predictions saved → %s", output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run inference with a registered churn model."
    )
    parser.add_argument(
        "--model_uri",
        type=str,
        required=True,
        help="MLflow model URI, e.g. 'models:/churn-predictor/latest'",
    )
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Path to input CSV file.",
    )
    args = parser.parse_args()
    main(model_uri=args.model_uri, input_path=args.input_path)
