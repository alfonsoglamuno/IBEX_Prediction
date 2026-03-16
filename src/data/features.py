"""
Feature engineering for IBEX35 forecasting.

Generates:
  - Lagged returns
  - Rolling statistics (mean, std, skew, min/max ratio)
  - Technical indicators (RSI, MACD, Bollinger Bands, ATR, ADX)
  - Volume features
  - Calendar features
  - Target variables (direction 1d/5d, log-return 5d, vol regime)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.config import get
from src.utils.logging import get_logger

log = get_logger(__name__)

# ── helpers ──────────────────────────────────────────────────────────────────

def _log_return(s: pd.Series, n: int = 1) -> pd.Series:
    return np.log(s / s.shift(n))


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


# ── main pipeline ─────────────────────────────────────────────────────────────

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Input:  raw OHLCV DataFrame (DatetimeIndex, columns: Open High Low Close Volume)
    Output: feature matrix + target columns, NaN rows trimmed.
    """
    cfg_lags = get("features.lags", [1, 2, 3, 5, 10, 21])
    cfg_wins = get("features.windows", [5, 10, 21, 63])
    use_vol  = get("features.include_volume", True)

    out = pd.DataFrame(index=df.index)
    c = df["Close"]
    h = df["High"]
    l = df["Low"]
    v = df["Volume"]

    # ── Lagged log returns ───────────────────────────────────────────────────
    log_ret = _log_return(c, 1)
    for lag in cfg_lags:
        out[f"ret_lag{lag}"] = log_ret.shift(lag - 1)  # lag-1 already, shift by lag-1 again → fully lagged

    # Simpler: compute daily log return, then shift
    daily_lr = _log_return(c, 1)
    out = pd.DataFrame(index=df.index)   # reset, rebuild cleanly

    for lag in cfg_lags:
        out[f"ret_lag{lag}"] = daily_lr.shift(lag)

    # ── Rolling statistics ───────────────────────────────────────────────────
    for w in cfg_wins:
        roll = daily_lr.rolling(w)
        out[f"ret_mean_{w}d"]  = roll.mean()
        out[f"ret_std_{w}d"]   = roll.std()
        out[f"ret_skew_{w}d"]  = roll.skew()
        out[f"hl_ratio_{w}d"]  = (h.rolling(w).max() - l.rolling(w).min()) / c.rolling(w).mean()

    # ── Technical indicators ─────────────────────────────────────────────────
    for p in [7, 14, 21]:
        out[f"rsi_{p}"] = _rsi(c, p)

    # MACD
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal    = macd_line.ewm(span=9, adjust=False).mean()
    out["macd"]        = macd_line
    out["macd_signal"] = signal
    out["macd_hist"]   = macd_line - signal

    # Bollinger Bands (20-day)
    boll_mid = c.rolling(20).mean()
    boll_std = c.rolling(20).std()
    out["boll_upper_dist"] = (c - (boll_mid + 2 * boll_std)) / c
    out["boll_lower_dist"] = (c - (boll_mid - 2 * boll_std)) / c
    out["boll_width"]      = (4 * boll_std) / boll_mid

    # ATR (normalised)
    for p in [7, 14]:
        out[f"atr_{p}_norm"] = _atr(h, l, c, p) / c

    # ── Price-level context ──────────────────────────────────────────────────
    for w in cfg_wins:
        out[f"dist_52w_high_{w}d"] = c / c.rolling(w).max() - 1
        out[f"dist_52w_low_{w}d"]  = c / c.rolling(w).min() - 1

    # ── Volume features ──────────────────────────────────────────────────────
    if use_vol and "Volume" in df.columns:
        v_z = (v - v.rolling(21).mean()) / v.rolling(21).std()
        out["vol_z21"]       = v_z
        out["vol_ratio_5_21"] = v.rolling(5).mean() / v.rolling(21).mean()
        out["price_vol"]     = daily_lr * v_z   # signed volume surprise

    # ── Calendar features ────────────────────────────────────────────────────
    out["dow"]          = df.index.dayofweek.astype(float)
    out["month"]        = df.index.month.astype(float)
    out["is_month_end"] = df.index.is_month_end.astype(float)

    # ── Volatility regime ────────────────────────────────────────────────────
    rv_21 = daily_lr.rolling(21).std() * np.sqrt(252)
    out["rv_21"] = rv_21

    # ── TARGET VARIABLES ─────────────────────────────────────────────────────
    # These are forward-looking — they must only be used as y, never as X
    fwd_1d  = -daily_lr.shift(-1)   # negate so positive = price goes up
    fwd_5d  = _log_return(c, 5).shift(-5)
    fwd_1d_actual = daily_lr.shift(-1)
    fwd_5d_actual = _log_return(c, 5).shift(-5)

    out["target_dir_1d"]    = (fwd_1d_actual > 0).astype(int)
    out["target_dir_5d"]    = (fwd_5d_actual > 0).astype(int)
    out["target_ret_5d"]    = fwd_5d_actual
    # Volatility bucket: high vol regime (top tercile of 21d RV)
    rv_q33, rv_q66 = rv_21.quantile(0.33), rv_21.quantile(0.66)
    out["target_vol_regime"] = pd.cut(rv_21, bins=[-np.inf, rv_q33, rv_q66, np.inf],
                                       labels=[0, 1, 2]).astype(float)

    # ── Drop rows that are NaN in any feature ────────────────────────────────
    feature_cols = [c for c in out.columns if not c.startswith("target_")]
    before = len(out)
    out.dropna(subset=feature_cols, inplace=True)
    log.info(f"Feature matrix: {len(out)} rows (dropped {before - len(out)} NaN rows from {before})")

    return out


def feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if not c.startswith("target_")]


def target_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("target_")]
