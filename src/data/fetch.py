"""
Download and cache IBEX35 OHLCV data from Yahoo Finance.
Uses parquet files for fast local caching.
"""
from __future__ import annotations

import hashlib
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import yfinance as yf

from src.utils.config import get, root
from src.utils.logging import get_logger

log = get_logger(__name__)


def _cache_path(ticker: str, start: str, end: str) -> Path:
    cache_dir = root() / get("data.cache_dir", "data/cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = hashlib.md5(f"{ticker}_{start}_{end}".encode()).hexdigest()[:8]
    return cache_dir / f"{ticker.replace('^', '')}_{start}_{end}_{key}.parquet"


def fetch_ohlcv(
    ticker: str | None = None,
    start: str | None = None,
    end: str | None = None,
    force: bool = False,
) -> pd.DataFrame:
    """
    Returns daily OHLCV DataFrame with a DatetimeIndex (timezone-naive, UTC).
    Caches to parquet so repeated calls are instant.
    """
    ticker = ticker or get("data.ticker", "^IBEX")
    start = start or get("data.start_date", "2010-01-01")
    end = end or get("data.end_date") or date.today().isoformat()

    cache = _cache_path(ticker, start, end)
    if cache.exists() and not force:
        log.info(f"Loading from cache: {cache}")
        df = pd.read_parquet(cache)
        return df

    log.info(f"Downloading {ticker} from {start} to {end}")
    raw = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)

    if raw.empty:
        raise ValueError(f"yfinance returned no data for {ticker} [{start} to {end}]")

    # Flatten MultiIndex columns if present (yfinance v0.2+)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.index.name = "Date"
    df.sort_index(inplace=True)
    df.dropna(subset=["Close"], inplace=True)

    df.to_parquet(cache)
    log.info(f"Saved {len(df)} rows to {cache}")
    return df


def load_raw(force: bool = False) -> pd.DataFrame:
    """Convenience wrapper using config defaults."""
    return fetch_ohlcv(force=force)
