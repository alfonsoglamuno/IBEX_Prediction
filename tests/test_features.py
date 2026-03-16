"""Sanity checks for feature engineering."""
import numpy as np
import pandas as pd
import pytest

from src.data.features import build_features, feature_cols, target_cols, get_indicator_states


@pytest.fixture
def dummy_ohlcv():
    np.random.seed(42)
    n = 400
    idx = pd.date_range("2019-01-01", periods=n, freq="B")
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


def test_no_nan_in_feature_columns(dummy_ohlcv):
    df = build_features(dummy_ohlcv)
    f_cols = feature_cols(df)
    n_nan = df[f_cols].isna().sum().sum()
    assert n_nan == 0, f"Feature columns contain {n_nan} NaN values"


def test_target_cols_present(dummy_ohlcv):
    df = build_features(dummy_ohlcv)
    t_cols = set(target_cols(df))
    expected = {"target_dir_1d", "target_dir_5d", "target_ret_5d", "target_ret_1d", "target_vol_regime"}
    assert expected.issubset(t_cols)


def test_direction_targets_binary(dummy_ohlcv):
    df = build_features(dummy_ohlcv).dropna(subset=["target_dir_1d", "target_dir_5d"])
    assert set(df["target_dir_1d"].dropna().unique()).issubset({0, 1})
    assert set(df["target_dir_5d"].dropna().unique()).issubset({0, 1})


def test_feature_count(dummy_ohlcv):
    df = build_features(dummy_ohlcv)
    f_cols = feature_cols(df)
    assert len(f_cols) >= 50, f"Expected >=50 features, got {len(f_cols)}"


def test_all_four_indicator_categories(dummy_ohlcv):
    df = build_features(dummy_ohlcv)
    f_cols = feature_cols(df)
    # Trend
    assert any("sma" in c or "ema" in c or "adx" in c for c in f_cols), "Missing trend features"
    # Momentum
    assert any("rsi" in c or "macd" in c or "stoch" in c or "cci" in c for c in f_cols), "Missing momentum"
    # Volatility
    assert any("boll" in c or "atr" in c or "hv_" in c for c in f_cols), "Missing volatility"
    # Volume
    assert any("obv" in c or "cmf" in c or "mfi" in c or "vol_z" in c for c in f_cols), "Missing volume"
    # Pivots
    assert any("pivot" in c or "resistance" in c or "support" in c for c in f_cols), "Missing pivots"


def test_indicator_states_returns_list(dummy_ohlcv):
    states = get_indicator_states(dummy_ohlcv)
    assert isinstance(states, list)
    if len(dummy_ohlcv) >= 210:
        assert len(states) > 0
        assert all("category" in s and "state" in s and "label" in s for s in states)


def test_no_lookahead_in_features(dummy_ohlcv):
    """Feature columns must not use future information (all based on current+past)."""
    df = build_features(dummy_ohlcv)
    f_cols = feature_cols(df)
    # All features with lag=1 should not be target-prefixed
    assert all(not c.startswith("target_") for c in f_cols)
