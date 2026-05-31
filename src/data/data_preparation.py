"""
Data Preparation Module
=======================
Loads the raw E-Commerce churn dataset, performs cleaning, encodes
categorical variables, and splits the data into train / validation / test
sets using a stratified split.

Usage
-----
    python src/data/data_preparation.py
"""

import logging
import os

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path: str = "conf/config.yaml") -> dict:
    """Load project configuration from YAML file.

    Args:
        config_path: Path to the configuration file.

    Returns:
        Dictionary containing configuration parameters.
    """
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_raw_data(raw_path: str) -> pd.DataFrame:
    """Load raw Excel data from disk.

    Automatically selects the sheet with the most rows when the
    workbook contains multiple sheets.

    Args:
        raw_path: Path to the raw .xlsx file.

    Returns:
        DataFrame with raw data.

    Raises:
        FileNotFoundError: If the raw file does not exist.
    """
    if not os.path.exists(raw_path):
        raise FileNotFoundError(
            f"Raw data not found at '{raw_path}'. "
            "Download the E-Commerce churn dataset and place it there."
        )

    # Read all sheets, then keep the largest one (the actual data sheet)
    all_sheets = pd.read_excel(raw_path, sheet_name=None)
    sheet_name, df = max(all_sheets.items(), key=lambda kv: len(kv[1]))
    logger.info("Using sheet '%s'", sheet_name)
    logger.info("Loaded raw data: %d rows, %d columns", *df.shape)
    logger.info("Columns: %s", df.columns.tolist())
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the raw dataframe.

    Steps:
    - Drop duplicate rows.
    - Fill numeric missing values with column median.
    - Fill categorical missing values with mode.

    Args:
        df: Raw DataFrame.

    Returns:
        Cleaned DataFrame.
    """
    before = len(df)
    df = df.drop_duplicates()
    logger.info("Dropped %d duplicate rows", before - len(df))

    num_cols = df.select_dtypes(include="number").columns
    cat_cols = df.select_dtypes(include="object").columns

    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    logger.info("Missing values after cleaning: %d", df.isnull().sum().sum())
    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Label-encode all object columns.

    Args:
        df: Cleaned DataFrame.

    Returns:
        DataFrame with encoded categorical columns.
    """
    le = LabelEncoder()
    cat_cols = df.select_dtypes(include="object").columns
    for col in cat_cols:
        df[col] = le.fit_transform(df[col].astype(str))
        logger.info("Encoded column: %s", col)
    return df


def split_data(
    df: pd.DataFrame,
    target: str,
    test_size: float,
    val_size: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split dataset into train, validation, and test sets (stratified).

    Args:
        df: Full cleaned and encoded DataFrame.
        target: Name of the target column.
        test_size: Fraction of data to reserve for test.
        val_size: Fraction of remaining data to reserve for validation.
        random_state: Random seed for reproducibility.

    Returns:
        Tuple of (train_df, val_df, test_df).
    """
    train_val, test = train_test_split(
        df,
        test_size=test_size,
        stratify=df[target],
        random_state=random_state,
    )
    val_fraction = val_size / (1 - test_size)
    train, val = train_test_split(
        train_val,
        test_size=val_fraction,
        stratify=train_val[target],
        random_state=random_state,
    )
    logger.info(
        "Split sizes -> train: %d | val: %d | test: %d",
        len(train),
        len(val),
        len(test),
    )
    return train, val, test


def main() -> None:
    """Main entry point for data preparation pipeline."""
    cfg = load_config()
    data_cfg = cfg["data"]
    feat_cfg = cfg["features"]

    df = load_raw_data(data_cfg["raw_path"])
    df = clean_data(df)
    df = encode_categoricals(df)

    # Drop columns not used for modelling
    drop_cols = [c for c in feat_cfg.get("drop_columns", []) if c in df.columns]
    df = df.drop(columns=drop_cols)
    logger.info("Dropped columns: %s", drop_cols)

    train, val, test = split_data(
        df,
        target=feat_cfg["target_column"],
        test_size=data_cfg["test_size"],
        val_size=data_cfg["val_size"],
        random_state=data_cfg["random_state"],
    )

    os.makedirs("data/processed", exist_ok=True)
    train.to_csv("data/processed/train.csv", index=False)
    val.to_csv("data/processed/val.csv", index=False)
    test.to_csv("data/processed/test.csv", index=False)
    logger.info("Saved train/val/test splits to data/processed/")


if __name__ == "__main__":
    main()
