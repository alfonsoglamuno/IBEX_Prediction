"""
IBEX35 Forecast REST API

Endpoints:
  GET  /health           — liveness check
  GET  /signal           — current UP/DOWN/NEUTRAL signal (latest_prediction.json)
  GET  /indicators       — current technical indicator states
  GET  /latest           — most recent feature vector
  GET  /history          — last N days of OHLCV + log-return
  GET  /results/{target} — walk-forward summary for a trained target
  POST /predict          — run fresh prediction (no SHAP, fast)
  POST /predict/shap     — run fresh prediction with SHAP drivers (slower)

Run with:
    uvicorn app.api:app --reload --port 8000
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from src.data.fetch import load_raw
from src.data.features import build_features, feature_cols, get_indicator_states
from src.utils.config import root
from src.utils.logging import get_logger

log = get_logger(__name__)

app = FastAPI(
    title="IBEX35 Forecast API",
    version="1.0.0",
    description=(
        "Walk-forward ML forecasting system for the IBEX35 index. "
        "Provides directional signals, SHAP explainability, and backtest results."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────────────────────────────────────
# In-memory feature cache (refreshed on startup)
# ─────────────────────────────────────────────────────────────────────────────
_CACHE: dict[str, Any] = {}


def _get_features() -> pd.DataFrame:
    if "df_feat" not in _CACHE:
        df_raw = load_raw()
        _CACHE["df_raw"]  = df_raw
        _CACHE["df_feat"] = build_features(df_raw)
    return _CACHE["df_feat"]


def _get_raw() -> pd.DataFrame:
    _get_features()   # ensures df_raw is also in cache
    return _CACHE["df_raw"]


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health", tags=["system"])
def health():
    """Liveness check — returns current server date."""
    return {"status": "ok", "date": date.today().isoformat()}


@app.get("/signal", tags=["prediction"])
def get_signal():
    """
    Return the latest cached market signal (UP/DOWN/NEUTRAL).

    Reads results/latest_prediction.json produced by `python predict.py`.
    Run `python predict.py --shap` to refresh.
    """
    path = root() / "results" / "latest_prediction.json"
    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail="No prediction found. Run `python predict.py` first.",
        )
    with open(path) as f:
        return json.load(f)


@app.get("/indicators", tags=["analysis"])
def get_indicators():
    """
    Return the current state of all technical indicators.

    Groups: Trend, Momentum, Volatility, Volume, Pivots.
    Each entry has: category, name, value, state, label.
    """
    df_raw = _get_raw()
    states = get_indicator_states(df_raw)
    if not states:
        raise HTTPException(status_code=503, detail="Insufficient data (need >210 trading days).")
    return {"date": df_raw.index[-1].isoformat(), "indicators": states}


@app.get("/latest", tags=["features"])
def latest_features(n: int = Query(1, ge=1, le=10, description="Number of recent rows")):
    """Return the most recent N feature rows (for inference display)."""
    df = _get_features()
    feat_c = feature_cols(df)
    rows = []
    for _, row in df[feat_c].tail(n).iterrows():
        rows.append({
            "date":     row.name.isoformat(),
            "features": {k: round(float(v), 6) for k, v in row.items() if pd.notna(v)},
        })
    return rows if n > 1 else rows[0]


@app.get("/history", tags=["market-data"])
def price_history(
    n: int = Query(252, ge=1, le=5000, description="Number of trading days"),
):
    """Return last N trading days of OHLCV + daily log-return."""
    df_raw = _get_raw()
    df_slice = df_raw.tail(n).copy()
    log_ret = np.log(df_slice["Close"] / df_slice["Close"].shift(1))
    df_slice["log_return"] = log_ret.round(6)
    records = (
        df_slice.reset_index()
        .rename(columns={"Date": "date", "index": "date"})
    )
    records["date"] = records.iloc[:, 0].astype(str)
    return records.to_dict(orient="records")


@app.get("/results/{target}", tags=["evaluation"])
def get_results(target: str):
    """
    Return walk-forward summary metrics for a trained target.

    Valid targets: target_dir_1d, target_dir_5d, target_ret_5d, target_vol_regime
    """
    valid = {"target_dir_1d", "target_dir_5d", "target_ret_5d", "target_vol_regime"}
    if target not in valid:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown target '{target}'. Valid options: {sorted(valid)}",
        )
    path = root() / "results" / f"{target}_summary.json"
    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"No results for '{target}'. Run `python train.py --target {target}`.",
        )
    with open(path) as f:
        return json.load(f)


@app.post("/predict", tags=["prediction"])
def run_predict(no_multiasset: bool = True):
    """
    Run a fresh prediction on the latest data row (no SHAP, fast ~2s).

    Returns signals for all available trained models.
    """
    import subprocess, sys
    cmd = [sys.executable, str(root() / "predict.py")]
    if no_multiasset:
        cmd.append("--no-multiasset")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if result.returncode != 0:
        raise HTTPException(status_code=500, detail=result.stderr[-500:])
    # Return the freshly generated prediction
    return get_signal()


@app.post("/predict/shap", tags=["prediction"])
def run_predict_shap(no_multiasset: bool = True):
    """
    Run a fresh prediction with SHAP top-driver analysis (~30s).

    Includes top 10 SHAP drivers per target in the response.
    """
    import subprocess, sys
    cmd = [sys.executable, str(root() / "predict.py"), "--shap"]
    if no_multiasset:
        cmd.append("--no-multiasset")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        raise HTTPException(status_code=500, detail=result.stderr[-500:])
    return get_signal()


@app.get("/summary", tags=["prediction"])
def get_summary(live_news: bool = True):
    """
    Return the cached operative summary (narrative + sentiment + media).

    Reads results/latest_summary.json if available; otherwise generates on-the-fly.
    live_news=true fetches current RSS headlines (adds ~10s on first call).
    """
    cached = root() / "results" / "latest_summary.json"
    if cached.exists() and not live_news:
        with open(cached) as f:
            return json.load(f)
    try:
        from app.summary import generate_summary, save_summary
        summary = generate_summary(include_live_news=live_news)
        save_summary(summary)
        return summary
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
