"""Unit tests for feature engineering functions."""

import pandas as pd
import pytest

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from features.feature_engineering import build_features


@pytest.fixture
def sample_df():
    return pd.DataFrame(
        {
            "Churn": [0, 1, 0],
            "DaySinceLastOrder": [5, 100, 30],
            "Tenure": [12, 1, 6],
            "OrderCount": [10, 1, 5],
            "SatisfactionScore": [4, 1, 3],
            "CouponUsed": [3, 0, 1],
            "CashbackAmount": [150.0, 10.0, 80.0],
            "HourSpendOnApp": [3, 0, 1],
        }
    )


def test_build_features_adds_recency_log(sample_df):
    result = build_features(sample_df)
    assert "recency_log" in result.columns


def test_build_features_adds_order_frequency(sample_df):
    result = build_features(sample_df)
    assert "order_frequency" in result.columns


def test_build_features_low_satisfaction_flag(sample_df):
    result = build_features(sample_df)
    # SatisfactionScore=1 → low_satisfaction=1
    assert result.loc[1, "low_satisfaction"] == 1
    # SatisfactionScore=4 → low_satisfaction=0
    assert result.loc[0, "low_satisfaction"] == 0


def test_build_features_is_idempotent(sample_df):
    result1 = build_features(sample_df)
    result2 = build_features(result1)
    assert result1.shape == result2.shape
