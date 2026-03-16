"""
FastAPI backend — serves latest predictions and backtest summary.

Run with:  uvicorn app.api:app --reload --port 8000
"""
from __future__ import annotations

import json
from pathlib import Path
from datetime import date

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.data.fetch import load_raw
from src.data.features import build_features, feature_cols
from src.utils.config import root
from src.utils.logging import get_logger

log = get_logger(__name__)
app = FastAPI(title="IBEX35 Forecast API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

_CACHE: dict = {}


def _load_features() -> pd.DataFrame:
    if "df" not in _CACHE:
        df_raw = load_raw()
        _CACHE["df"] = build_features(df_raw)
    return _CACHE["df"]


@app.get("/health")
def health():
    return {"status": "ok", "date": date.today().isoformat()}


@app.get("/latest")
def latest_features():
    """Return the most recent feature row (for inference display)."""
    df = _load_features()
    latest = df.iloc[-1]
    feat_c = feature_cols(df)
    return {
        "date": latest.name.isoformat(),
        "features": {k: round(float(latest[k]), 6) for k in feat_c if pd.notna(latest[k])},
    }


@app.get("/results/{target}")
def get_results(target: str):
    """Return walk-forward summary for a given target."""
    valid = {"target_dir_1d", "target_dir_5d", "target_ret_5d", "target_vol_regime"}
    if target not in valid:
        raise HTTPException(status_code=400, detail=f"Unknown target. Choose from {valid}")
    path = root() / "results" / f"{target}_summary.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"No results found. Run train.py first.")
    with open(path) as f:
        return json.load(f)


@app.get("/history")
def price_history(n: int = 252):
    """Return last n trading days of OHLCV + daily returns."""
    df_raw = load_raw()
    df_slice = df_raw.tail(n).copy()
    df_slice["log_return"] = (df_slice["Close"] / df_slice["Close"].shift(1)).apply(
        lambda x: round(float(x), 6) if pd.notna(x) else None
    )
    records = df_slice.reset_index().rename(columns={"Date": "date"})
    return records.to_dict(orient="records")
