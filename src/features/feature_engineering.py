"""
Feature Engineering Module
===========================
Builds churn-predictive features from the processed train/val/test splits.
Key feature families:

- **RFM indicators**: Recency (days since last order), Frequency
  (order count), Monetary (order value).
- **Engagement decay**: hours on app, days since last login.
- **Satisfaction signals**: complain flag, satisfaction score.
- **Behaviour ratios**: coupon usage rate, cashback per order.

The same transformations are applied consistently across all splits to
prevent data leakage.

Usage
-----
    python src/features/feature_engineering.py
"""

import logging
import os

import numpy as np
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


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer churn-predictive features from a processed split.

    This function is idempotent: running it twice on the same DataFrame
    produces the same result. New columns are added; no existing columns
    are modified.

    Args:
        df: Processed DataFrame (output of data_preparation.py).

    Returns:
        DataFrame enriched with engineered features.
    """
    df = df.copy()

    # --- Recency / engagement decay ---
    if "DaySinceLastOrder" in df.columns:
        # Customers who haven't ordered recently are higher risk
        df["recency_log"] = np.log1p(df["DaySinceLastOrder"])

    if "DaysSinceLastLogin" in df.columns:
        df["login_recency_log"] = np.log1p(df["DaysSinceLastLogin"])

    # --- Frequency & monetary ---
    if "OrderCount" in df.columns and "Tenure" in df.columns:
        # Orders per month of tenure (avoid division by zero)
        df["order_frequency"] = df["OrderCount"] / (df["Tenure"] + 1)

    if "OrderAmountHikeFromlastYear" in df.columns:
        # Negative or zero hike signals disengagement
        df["spend_declining"] = (df["OrderAmountHikeFromlastYear"] <= 0).astype(int)

    # --- Satisfaction & complaints ---
    if "SatisfactionScore" in df.columns:
        df["low_satisfaction"] = (df["SatisfactionScore"] <= 2).astype(int)

    # --- Coupon & cashback behaviour ---
    if "CouponUsed" in df.columns and "OrderCount" in df.columns:
        df["coupon_rate"] = df["CouponUsed"] / (df["OrderCount"] + 1)

    if "CashbackAmount" in df.columns and "OrderCount" in df.columns:
        df["cashback_per_order"] = df["CashbackAmount"] / (df["OrderCount"] + 1)

    # --- App & device engagement ---
    if "HourSpendOnApp" in df.columns:
        df["low_app_usage"] = (df["HourSpendOnApp"] < 2).astype(int)

    logger.info("Feature engineering done. Shape: %d rows × %d cols", *df.shape)
    return df


def main() -> None:
    """Main entry point for the feature engineering pipeline."""
    load_config()  # validate config is readable

    os.makedirs("data/processed", exist_ok=True)

    for split in ("train", "val", "test"):
        path = f"data/processed/{split}.csv"
        if not os.path.exists(path):
            logger.warning("%s not found — run data_preparation.py first.", path)
            continue
        df = pd.read_csv(path)
        df = build_features(df)
        out_path = f"data/processed/{split}_features.csv"
        df.to_csv(out_path, index=False)
        logger.info("Saved engineered features → %s", out_path)


if __name__ == "__main__":
    main()
