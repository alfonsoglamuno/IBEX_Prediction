"""
Walk-forward validation and realistic backtesting.

Protocol:
  - Strictly time-ordered (no shuffling)
  - Expanding train window
  - Configurable step and validation window
  - Daily PnL uses 1-day forward return (not 5-day)
  - Transaction costs on each position change
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.utils.config import get
from src.utils.logging import get_logger
from src.evaluation.metrics import classification_report, regression_report, backtest_metrics

log = get_logger(__name__)


@dataclass
class WalkForwardResult:
    fold_metrics: list[dict] = field(default_factory=list)
    all_predictions: pd.DataFrame | None = None
    summary: dict = field(default_factory=dict)


def walk_forward_cv(
    df_feat: pd.DataFrame,
    target: str,
    make_model: Callable,
    task: str = "classification",
    initial_train_months: int | None = None,
    step_months: int | None = None,
    val_months: int | None = None,
    min_confidence: float | None = None,
    transaction_cost_bps: float | None = None,
    verbose: bool = True,
) -> WalkForwardResult:
    """
    Expanding-window walk-forward cross-validation.

    For classification:
      - ML metrics: accuracy, F1, ROC-AUC, PR-AUC
      - Economic metrics: Sharpe, max-drawdown, Calmar, hit-ratio
        (PnL uses 1-day forward return from 'target_ret_1d')

    For regression:
      - ML metrics: MAE, RMSE, R², IC
    """
    cfg_wf   = get("walk_forward", {})
    init_m   = initial_train_months or cfg_wf.get("initial_train_months", 36)
    step_m   = step_months or cfg_wf.get("step_months", 3)
    val_m    = val_months  or cfg_wf.get("val_months", 3)
    tc_bps   = transaction_cost_bps if transaction_cost_bps is not None \
               else get("backtest.transaction_cost_bps", 10)
    min_conf = min_confidence if min_confidence is not None \
               else get("backtest.min_confidence", 0.55)

    feat_cols = [col for col in df_feat.columns if not col.startswith("target_")]
    X     = df_feat[feat_cols].values
    y     = df_feat[target].values
    dates = df_feat.index

    fold_starts = pd.date_range(
        start=dates[0] + pd.DateOffset(months=init_m),
        end  =dates[-1] - pd.DateOffset(months=val_m),
        freq =f"{step_m}MS",
    )

    result   = WalkForwardResult()
    all_pred = []
    iterable = tqdm(fold_starts, desc=f"Walk-forward [{target}]") if verbose else fold_starts

    for fold_start in iterable:
        fold_end   = fold_start + pd.DateOffset(months=val_m)
        train_mask = dates < fold_start
        val_mask   = (dates >= fold_start) & (dates < fold_end)

        if train_mask.sum() < 50 or val_mask.sum() < 5:
            continue

        X_tr, y_tr = X[train_mask], y[train_mask]
        X_va, y_va = X[val_mask],   y[val_mask]
        dates_va   = dates[val_mask]

        model = make_model()

        # ── Fit ───────────────────────────────────────────────────────────────
        if task == "classification":
            try:
                model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
            except TypeError:
                model.fit(X_tr, y_tr)

            y_prob = _get_proba(model, X_va)
            y_pred = (y_prob >= 0.5).astype(int)
            metrics = classification_report(y_va, y_pred, y_prob)

            # ── Backtest: use actual 1d forward returns as daily PnL ──────────
            # Position: 1 if P(up) >= min_conf, 0 otherwise (long-only / flat)
            positions  = np.where(y_prob >= min_conf, 1.0, 0.0)
            if "target_ret_1d" in df_feat.columns:
                bm_daily = df_feat.loc[dates_va, "target_ret_1d"].values
            else:
                # Fallback: approximate from 5d
                bm_daily = df_feat.loc[dates_va, "target_ret_5d"].values / 5.0 \
                           if "target_ret_5d" in df_feat.columns \
                           else np.zeros(len(dates_va))

            strat_ret = _apply_costs(positions, bm_daily, tc_bps)
            bt = backtest_metrics(
                pd.Series(strat_ret, index=dates_va),
                pd.Series(bm_daily,  index=dates_va),
            )
            metrics.update({f"bt_{k}": v for k, v in bt.items()})

        else:
            try:
                model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
            except TypeError:
                model.fit(X_tr, y_tr)
            y_pred = model.predict(X_va)
            y_prob = None
            metrics = regression_report(y_va, y_pred)

        metrics.update({
            "fold_start": fold_start,
            "n_train":    int(train_mask.sum()),
            "n_val":      int(val_mask.sum()),
        })
        result.fold_metrics.append(metrics)

        fold_df = pd.DataFrame({"y_true": y_va, "y_pred": y_pred}, index=dates_va)
        if y_prob is not None:
            fold_df["y_prob"] = y_prob
        all_pred.append(fold_df)

    result.all_predictions = pd.concat(all_pred) if all_pred else None

    if result.fold_metrics:
        num_keys = [k for k, v in result.fold_metrics[0].items() if isinstance(v, float)]
        result.summary = {
            k: float(np.mean([f[k] for f in result.fold_metrics if k in f]))
            for k in num_keys
        }

    log.info(f"Walk-forward done: {len(result.fold_metrics)} folds")
    return result


# ── helpers ───────────────────────────────────────────────────────────────────

def _get_proba(model, X: np.ndarray) -> np.ndarray:
    """Unified probability extraction for sklearn-compatible and custom models."""
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        # sklearn: shape (n, 2); custom LSTM trainer: shape (n,)
        return proba[:, 1] if proba.ndim == 2 else proba
    raise AttributeError(f"Model {type(model).__name__} has no predict_proba method")


def _apply_costs(positions: np.ndarray, returns: np.ndarray,
                 cost_bps: float) -> np.ndarray:
    """Apply one-way transaction cost on each position change."""
    cost     = cost_bps / 10_000
    strategy = positions * returns
    trades   = np.abs(np.diff(positions, prepend=0.0))
    strategy -= trades * cost
    return strategy
