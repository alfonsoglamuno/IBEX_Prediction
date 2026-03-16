"""Basic sanity checks for feature engineering."""
import numpy as np
import pandas as pd
import pytest

from src.data.features import build_features, feature_cols, target_cols


@pytest.fixture
def dummy_ohlcv():
    np.random.seed(42)
    n = 300
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    close = 8000 + np.cumsum(np.random.randn(n) * 50)
    df = pd.DataFrame({
        "Open":   close * (1 - 0.002),
        "High":   close * 1.005,
        "Low":    close * 0.995,
        "Close":  close,
        "Volume": np.random.randint(1_000_000, 5_000_000, n).astype(float),
    }, index=idx)
    return df


def test_build_features_returns_dataframe(dummy_ohlcv):
    df = build_features(dummy_ohlcv)
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0


def test_no_nan_in_features(dummy_ohlcv):
    df = build_features(dummy_ohlcv)
    f_cols = feature_cols(df)
    assert df[f_cols].isna().sum().sum() == 0, "Feature columns must not contain NaN"


def test_target_cols_present(dummy_ohlcv):
    df = build_features(dummy_ohlcv)
    t_cols = target_cols(df)
    expected = {"target_dir_1d", "target_dir_5d", "target_ret_5d", "target_vol_regime"}
    assert expected.issubset(set(t_cols))


def test_direction_targets_are_binary(dummy_ohlcv):
    df = build_features(dummy_ohlcv).dropna(subset=["target_dir_1d", "target_dir_5d"])
    assert set(df["target_dir_1d"].unique()).issubset({0, 1})
    assert set(df["target_dir_5d"].unique()).issubset({0, 1})


def test_feature_count_reasonable(dummy_ohlcv):
    df = build_features(dummy_ohlcv)
    f_cols = feature_cols(df)
    assert len(f_cols) >= 20, f"Expected at least 20 features, got {len(f_cols)}"
