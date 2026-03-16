"""
Walk-forward validation and realistic backtesting.

Walk-forward protocol:
  - No random shuffling
  - Time-ordered folds
  - Retrain on each expanding or rolling window
  - Economic metrics computed alongside ML metrics
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

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
    all_predictions: pd.DataFrame | None = None   # index=Date, cols=y_true/y_pred/y_prob
    summary: dict = field(default_factory=dict)


def walk_forward_cv(
    df_feat: pd.DataFrame,
    target: str,
    make_model: Callable,
    task: str = "classification",        # "classification" | "regression"
    initial_train_months: int | None = None,
    step_months: int | None = None,
    val_months: int | None = None,
    min_confidence: float | None = None,
    transaction_cost_bps: float | None = None,
    verbose: bool = True,
) -> WalkForwardResult:
    """
    Performs expanding-window walk-forward cross-validation.

    Parameters
    ----------
    df_feat : full feature DataFrame (including target column)
    target  : name of the target column
    make_model : zero-arg factory that returns an unfitted model
    task    : "classification" or "regression"
    """
    cfg_wf  = get("walk_forward", {})
    init_m  = initial_train_months or cfg_wf.get("initial_train_months", 36)
    step_m  = step_months or cfg_wf.get("step_months", 3)
    val_m   = val_months  or cfg_wf.get("val_months", 3)
    tc_bps  = transaction_cost_bps if transaction_cost_bps is not None \
              else get("backtest.transaction_cost_bps", 10)
    min_conf = min_confidence if min_confidence is not None \
               else get("backtest.min_confidence", 0.55)

    feature_cols = [c for c in df_feat.columns if not c.startswith("target_")]
    X = df_feat[feature_cols].values
    y = df_feat[target].values
    dates = df_feat.index

    # Build fold boundaries (month-based)
    first_date = dates[0]
    last_date  = dates[-1]

    fold_starts = pd.date_range(
        start=first_date + pd.DateOffset(months=init_m),
        end=last_date   - pd.DateOffset(months=val_m),
        freq=f"{step_m}MS",
    )

    result = WalkForwardResult()
    all_preds = []

    iterable = tqdm(fold_starts, desc="Walk-forward folds") if verbose else fold_starts

    for fold_val_start in iterable:
        fold_val_end = fold_val_start + pd.DateOffset(months=val_m)

        train_mask = dates < fold_val_start
        val_mask   = (dates >= fold_val_start) & (dates < fold_val_end)

        if train_mask.sum() < 50 or val_mask.sum() < 5:
            continue

        X_tr, y_tr = X[train_mask], y[train_mask]
        X_va, y_va = X[val_mask],   y[val_mask]
        dates_va   = dates[val_mask]

        model = make_model()

        if task == "classification":
            # XGBoost needs eval_set; handle gracefully
            try:
                model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
            except TypeError:
                model.fit(X_tr, y_tr)

            y_prob = _get_proba(model, X_va)
            y_pred = (y_prob >= 0.5).astype(int)

            metrics = classification_report(y_va, y_pred, y_prob)

            # Simple backtest: go long when P(up) > min_conf, flat otherwise
            # Transaction cost applied when position changes
            positions = np.where(y_prob >= min_conf, 1.0, 0.0)
            daily_ret = df_feat.loc[dates_va, "target_ret_5d"] \
                        if "target_ret_5d" in df_feat.columns \
                        else pd.Series(np.zeros(len(dates_va)), index=dates_va)
            strategy_ret = _apply_costs(positions, daily_ret.values, tc_bps)
            bm_ret       = daily_ret.values
            bt = backtest_metrics(
                pd.Series(strategy_ret, index=dates_va),
                pd.Series(bm_ret, index=dates_va),
            )
            metrics.update({f"bt_{k}": v for k, v in bt.items()})

        else:  # regression
            try:
                model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
            except TypeError:
                model.fit(X_tr, y_tr)
            y_pred = model.predict(X_va)
            y_prob = None
            metrics = regression_report(y_va, y_pred)

        metrics["fold_start"] = fold_val_start
        metrics["n_train"]    = int(train_mask.sum())
        metrics["n_val"]      = int(val_mask.sum())
        result.fold_metrics.append(metrics)

        fold_df = pd.DataFrame({
            "y_true": y_va,
            "y_pred": y_pred,
        }, index=dates_va)
        if y_prob is not None:
            fold_df["y_prob"] = y_prob
        all_preds.append(fold_df)

    result.all_predictions = pd.concat(all_preds) if all_preds else None

    # Summary: mean across folds
    numeric_keys = [k for k in result.fold_metrics[0] if isinstance(result.fold_metrics[0][k], float)] \
                   if result.fold_metrics else []
    result.summary = {k: np.mean([f[k] for f in result.fold_metrics if k in f])
                      for k in numeric_keys}

    log.info(f"Walk-forward done: {len(result.fold_metrics)} folds")
    return result


# ── helpers ───────────────────────────────────────────────────────────────────

def _get_proba(model, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    elif hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    else:
        # LSTM or custom wrapper
        return model.predict_proba(X)


def _apply_costs(positions: np.ndarray, returns: np.ndarray,
                 cost_bps: float) -> np.ndarray:
    """Apply one-way transaction cost on each position change."""
    cost = cost_bps / 10_000
    strategy = positions * returns
    trades = np.abs(np.diff(positions, prepend=0.0))
    strategy -= trades * cost
    return strategy
