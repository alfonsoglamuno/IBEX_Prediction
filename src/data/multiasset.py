"""
Multi-asset context features for IBEX35 forecasting.

Fetches correlated global assets and computes lagged returns to use as
additional predictors. All features are strictly lagged to avoid lookahead.

Assets:
  - S&P 500    (^GSPC) — global risk sentiment
  - DAX        (^GDAXI) — European market proxy
  - EUR/USD    (EURUSD=X) — Euro strength (Spain is EUR-denominated)
  - VIX        (^VIX) — fear / implied volatility
  - Brent Oil  (BZ=F) — energy prices (Spain energy costs)
  - US 10y     (^TNX) — macro rates / cost of capital
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

_ASSET_MAP = {
    "sp500":  "^GSPC",
    "dax":    "^GDAXI",
    "eurusd": "EURUSD=X",
    "vix":    "^VIX",
    "brent":  "BZ=F",
    "usbond": "^TNX",
}


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

    for name, ticker in tickers.items():
        s = _fetch_one(ticker, name, start, end, force)
        if s is None:
            continue

        # Align to IBEX trading calendar using forward-fill
        s = s.reindex(ibex_index, method="ffill")
        lr = np.log(s / s.shift(1))

        # Level feature for VIX (level matters, not just return)
        if name == "vix":
            vix_roll = s.rolling(21).mean()
            out["vix_level"]  = s
            out["vix_z21"]    = (s - vix_roll) / (s.rolling(21).std() + 1e-9)

        # Lagged returns
        for lag in lags:
            out[f"{name}_ret_lag{lag}"] = lr.shift(lag)

    if out.empty:
        log.warning("No multi-asset features computed — check internet / tickers")
    else:
        log.info(f"Multi-asset: {out.shape[1]} features for {len(out)} rows")

    return out
