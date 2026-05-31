"""
Explainable AI (XAI) Module — Part 3
======================================
Implements SHAP (SHapley Additive exPlanations) to explain the
churn prediction model trained in Part 2.

All required plots are generated and saved to ``outputs/shap/``:

- Summary plot (bar) for feature importance
- Summary plot (beeswarm) for all data points
- Summary plot per class
- Waterfall plot for a single prediction
- Force plot for a single prediction
- Mean SHAP bar plot
- Beeswarm plot
- Dependence plots for top features

Usage
-----
    python src/models/explain.py

Requirements
------------
    shap, matplotlib, mlflow, xgboost must be installed.
    Run train.py first so a model exists in the MLflow registry.
"""

import logging
import os

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving figures
import matplotlib.pyplot as plt
import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
import shap
import yaml

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = "outputs/shap"


def load_config(config_path: str = "conf/config.yaml") -> dict:
    """Load project configuration.

    Args:
        config_path: Path to YAML config file.

    Returns:
        Configuration dictionary.
    """
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_test_data(target: str) -> tuple[pd.DataFrame, pd.Series]:
    """Load the test feature set.

    Args:
        target: Name of the target column.

    Returns:
        Tuple of (X_test DataFrame, y_test Series).
    """
    path = "data/processed/test_features.csv"
    df = pd.read_csv(path)
    X = df.drop(columns=[target])
    y = df[target]
    logger.info("Loaded test data: %d rows, %d features", *X.shape)
    return X, y


def load_xgboost_model(model_uri: str):
    """Load XGBoost model from MLflow registry.

    Args:
        model_uri: MLflow model URI.

    Returns:
        Native XGBoost booster object.
    """
    logger.info("Loading model from: %s", model_uri)
    model = mlflow.xgboost.load_model(model_uri)
    return model


def save_fig(name: str) -> None:
    """Save current matplotlib figure to OUTPUT_DIR.

    Args:
        name: Filename without extension.
    """
    path = os.path.join(OUTPUT_DIR, f"{name}.png")
    plt.savefig(path, bbox_inches="tight", dpi=150)
    plt.close()
    logger.info("Saved → %s", path)


def run_shap_analysis(model, X_test: pd.DataFrame, y_test: pd.Series) -> None:
    """Run full SHAP analysis and generate all required plots.

    Plots generated:
    - summary_bar: mean |SHAP| bar chart (feature importance)
    - summary_beeswarm: beeswarm over all test points
    - summary_class0 / summary_class1: per-class summaries
    - waterfall_point0: waterfall for the first test instance
    - force_point0: force plot for the first test instance
    - mean_shap_bar: mean absolute SHAP values as bar chart
    - beeswarm: beeswarm plot (alias, required by Part 3 spec)
    - dependence_<feature>: dependence plots for top 3 features

    Args:
        model: Trained XGBoost model.
        X_test: Test feature DataFrame.
        y_test: Test target Series.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    logger.info("Building TreeExplainer and computing SHAP values...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test)

    # shap_values.values shape: (n_samples, n_features)
    sv = shap_values.values
    feature_names = X_test.columns.tolist()

    # ------------------------------------------------------------------ #
    # 1. Summary bar plot — mean |SHAP| feature importance
    # ------------------------------------------------------------------ #
    logger.info("Generating summary bar plot...")
    plt.figure(figsize=(10, 6))
    shap.summary_plot(sv, X_test, plot_type="bar", show=False)
    plt.title("Mean |SHAP| — Feature Importance")
    save_fig("summary_bar")

    # ------------------------------------------------------------------ #
    # 2. Beeswarm summary plot — all data points
    # ------------------------------------------------------------------ #
    logger.info("Generating beeswarm summary plot...")
    plt.figure(figsize=(10, 8))
    shap.summary_plot(sv, X_test, show=False)
    plt.title("SHAP Beeswarm — All Test Points")
    save_fig("summary_beeswarm")

    # ------------------------------------------------------------------ #
    # 3. Per-class summary plots
    # ------------------------------------------------------------------ #
    for cls, label in [(0, "No Churn"), (1, "Churn")]:
        logger.info("Generating summary plot for class %d (%s)...", cls, label)
        mask = y_test.values == cls
        if mask.sum() == 0:
            continue
        plt.figure(figsize=(10, 6))
        shap.summary_plot(sv[mask], X_test[mask], plot_type="bar", show=False)
        plt.title(f"Mean |SHAP| — Class {cls}: {label}")
        save_fig(f"summary_class{cls}")

    # ------------------------------------------------------------------ #
    # 4. Waterfall plot — single data point (index 0)
    # ------------------------------------------------------------------ #
    logger.info("Generating waterfall plot for point 0...")
    plt.figure(figsize=(10, 6))
    shap.plots.waterfall(shap_values[0], show=False)
    plt.title("Waterfall Plot — Customer 0")
    save_fig("waterfall_point0")

    # ------------------------------------------------------------------ #
    # 5. Force plot — single data point (index 0)
    # ------------------------------------------------------------------ #
    logger.info("Generating force plot for point 0...")
    force_html = shap.force_plot(
        explainer.expected_value,
        sv[0],
        X_test.iloc[0],
        show=False,
        matplotlib=False,
    )
    html_path = os.path.join(OUTPUT_DIR, "force_point0.html")
    shap.save_html(html_path, force_html)
    logger.info("Saved → %s", html_path)

    # Force plot as matplotlib figure
    plt.figure(figsize=(14, 3))
    shap.force_plot(
        explainer.expected_value,
        sv[0],
        X_test.iloc[0],
        matplotlib=True,
        show=False,
    )
    save_fig("force_point0")

    # ------------------------------------------------------------------ #
    # 6. Mean SHAP bar plot (explicit, required by spec)
    # ------------------------------------------------------------------ #
    logger.info("Generating mean SHAP bar plot...")
    mean_shap = np.abs(sv).mean(axis=0)
    sorted_idx = np.argsort(mean_shap)[::-1][:15]
    plt.figure(figsize=(10, 6))
    plt.barh(
        [feature_names[i] for i in sorted_idx[::-1]],
        mean_shap[sorted_idx[::-1]],
        color="steelblue",
    )
    plt.xlabel("Mean |SHAP Value|")
    plt.title("Mean SHAP — Top 15 Features")
    plt.tight_layout()
    save_fig("mean_shap_bar")

    # ------------------------------------------------------------------ #
    # 7. Beeswarm plot (explicit shap.plots.beeswarm)
    # ------------------------------------------------------------------ #
    logger.info("Generating beeswarm plot...")
    plt.figure(figsize=(10, 8))
    shap.plots.beeswarm(shap_values, show=False)
    plt.title("SHAP Beeswarm Plot")
    save_fig("beeswarm")

    # ------------------------------------------------------------------ #
    # 8. Dependence plots — top 3 most important features
    # ------------------------------------------------------------------ #
    top3_features = [feature_names[i] for i in np.argsort(mean_shap)[::-1][:3]]
    for feat in top3_features:
        logger.info("Generating dependence plot for '%s'...", feat)
        plt.figure(figsize=(8, 5))
        shap.dependence_plot(feat, sv, X_test, show=False)
        plt.title(f"SHAP Dependence Plot — {feat}")
        save_fig(f"dependence_{feat}")

    logger.info("All SHAP plots saved to '%s/'", OUTPUT_DIR)


def main() -> None:
    """Main entry point for SHAP explainability pipeline."""
    cfg = load_config()
    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])

    model_uri = f"models:/{cfg['mlflow']['model_name']}/latest"
    model = load_xgboost_model(model_uri)

    X_test, y_test = load_test_data(cfg["features"]["target_column"])

    run_shap_analysis(model, X_test, y_test)


if __name__ == "__main__":
    main()