"""
Multi-asset context features for IBEX35 forecasting.

Fetches correlated global assets and computes lagged returns as additional
predictors. All features are strictly lagged (shift ≥ 1) to avoid lookahead.

Asset groups:
  Euro-area benchmark:
    EURO STOXX 50 (^STOXX50E) — most direct regional benchmark for IBEX
  Global macro:
    S&P 500  (^GSPC)    — global risk-on/off
    DAX      (^GDAXI)   — German / euro proxy
    EUR/USD  (EURUSD=X) — Euro strength
    VIX      (^VIX)     — US implied vol / fear
    Brent    (BZ=F)     — energy / Spain energy exposure
    US 10y   (^TNX)     — macro rates / discount rate

Literature basis:
  EURO STOXX 50 returns have a strong negative correlation with IBEX volatility;
  using them as lagged regional factors (not contemporaneous) avoids lookahead
  while capturing euro-area regime information useful for IBEX prediction.
  (STOXX white paper; Giantsidi & Tarantola 2025 deep-learning review)
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from src.utils.config import get, root
from src.utils.logging import get_logger

log = get_logger(__name__)

# Default asset map — overridden by config.yaml multiasset.tickers
_ASSET_MAP = {
    "sp500":   "^GSPC",
    "dax":     "^GDAXI",
    "stoxx50": "^STOXX50E",
    "eurusd":  "EURUSD=X",
    "vix":     "^VIX",
    "brent":   "BZ=F",
    "usbond":  "^TNX",
}

# Assets that get level + z-score treatment (not just returns)
_LEVEL_ASSETS = {"vix"}


def _cache_path(name: str, start: str, end: str) -> Path:
    cache_dir = root() / get("data.cache_dir", "data/cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = hashlib.md5(f"{name}_{start}_{end}".encode()).hexdigest()[:6]
    return cache_dir / f"multiasset_{name}_{key}.parquet"


def _fetch_one(ticker: str, name: str, start: str, end: str,
               force: bool = False) -> pd.Series | None:
    cache = _cache_path(name, start, end)
    if cache.exists() and not force:
        return pd.read_parquet(cache)["close"]
    try:
        raw = yf.download(ticker, start=start, end=end,
                          auto_adjust=True, progress=False)
        if raw.empty:
            log.warning(f"No data for {ticker}")
            return None
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        s = raw["Close"].copy()
        s.index = pd.to_datetime(s.index).tz_localize(None)
        s.name = name
        pd.DataFrame({"close": s}).to_parquet(cache)
        return s
    except Exception as e:
        log.warning(f"Failed to fetch {ticker}: {e}")
        return None


def fetch_multiasset_features(
    ibex_index: pd.DatetimeIndex,
    start: str | None = None,
    end: str | None = None,
    force: bool = False,
) -> pd.DataFrame:
    """
    Returns a DataFrame aligned to ibex_index with lagged return features
    for each configured correlated asset.
    """
    if not get("multiasset.enabled", True):
        return pd.DataFrame(index=ibex_index)

    from datetime import date
    start = start or get("data.start_date", "2007-01-01")
    end   = end   or get("data.end_date") or date.today().isoformat()
    lags  = get("multiasset.lags", [1, 2, 5])
    tickers = get("multiasset.tickers", _ASSET_MAP)

    out = pd.DataFrame(index=ibex_index)

    level_assets = set(get("multiasset.level_assets", list(_LEVEL_ASSETS)))

    # IBEX close for relative-strength features
    ibex_close: pd.Series | None = None
    try:
        from src.data.fetch import load_raw
        ibex_close = load_raw()["Close"].reindex(ibex_index, method="ffill")
    except Exception:
        pass

    for name, ticker in tickers.items():
        s = _fetch_one(ticker, name, start, end, force)
        if s is None:
            continue

        # Align to IBEX trading calendar using forward-fill
        s = s.reindex(ibex_index, method="ffill")
        lr = np.log(s / s.shift(1))

        # Level + z-score for implied-vol assets (VIX, VSTOXX)
        if name in level_assets:
            roll21 = s.rolling(21).mean()
            out[f"{name}_level"] = s.shift(1)               # lag-1 level
            out[f"{name}_z21"]   = ((s - roll21) / (s.rolling(21).std() + 1e-9)).shift(1)

        # Lagged returns
        for lag in lags:
            out[f"{name}_ret_lag{lag}"] = lr.shift(lag)

    # Relative-strength feature: IBEX vs EURO STOXX 50
    # Captures whether IBEX is outperforming or underperforming its regional benchmark
    if ibex_close is not None and "stoxx50" in tickers:
        stoxx_s = _fetch_one(tickers["stoxx50"], "stoxx50", start, end, force)
        if stoxx_s is not None:
            stoxx_s = stoxx_s.reindex(ibex_index, method="ffill")
            rs = np.log(ibex_close / (stoxx_s + 1e-9))   # log ratio
            out["ibex_vs_stoxx_rs5"]  = rs.diff(5).shift(1)   # 5d relative momentum
            out["ibex_vs_stoxx_rs21"] = rs.diff(21).shift(1)  # 21d relative momentum

    if out.empty:
        log.warning("No multi-asset features computed — check internet / tickers")
    else:
        log.info(f"Multi-asset: {out.shape[1]} features for {len(out)} rows")

    return out
