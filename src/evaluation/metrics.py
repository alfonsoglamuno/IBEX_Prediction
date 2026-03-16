"""
ML metrics + economic/backtest metrics for IBEX35 forecasting evaluation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    f1_score, roc_auc_score, average_precision_score,
    mean_absolute_error,
)


# ── Classification ────────────────────────────────────────────────────────────

def classification_report(y_true: np.ndarray, y_pred: np.ndarray,
                           y_prob: np.ndarray | None = None) -> dict:
    out = {
        "accuracy":          accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "f1":                f1_score(y_true, y_pred, zero_division=0),
        "f1_macro":          f1_score(y_true, y_pred, average="macro", zero_division=0),
    }
    if y_prob is not None:
        out["roc_auc"]  = roc_auc_score(y_true, y_prob)
        out["pr_auc"]   = average_precision_score(y_true, y_prob)
    return out


# ── Regression ────────────────────────────────────────────────────────────────

def regression_report(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "mae":  mean_absolute_error(y_true, y_pred),
        "rmse": np.sqrt(np.mean((y_true - y_pred) ** 2)),
        "r2":   1 - np.sum((y_true - y_pred) ** 2) / (np.sum((y_true - np.mean(y_true)) ** 2) + 1e-12),
        "ic":   np.corrcoef(y_true, y_pred)[0, 1],   # information coefficient
    }


# ── Economic / backtest ───────────────────────────────────────────────────────

def backtest_metrics(returns: pd.Series, benchmark: pd.Series | None = None) -> dict:
    """
    returns: daily strategy return series (already net of costs)
    benchmark: optional buy-and-hold return series for the same period
    """
    ann   = 252
    total = (1 + returns).prod() - 1
    ann_r = (1 + returns).prod() ** (ann / len(returns)) - 1
    ann_v = returns.std() * np.sqrt(ann)
    sharpe = ann_r / (ann_v + 1e-12)

    # Max drawdown
    cum = (1 + returns).cumprod()
    roll_max = cum.cummax()
    dd = (cum - roll_max) / (roll_max + 1e-12)
    max_dd = dd.min()

    # Calmar
    calmar = ann_r / (abs(max_dd) + 1e-12)

    # Hit ratio (fraction of winning days)
    hit = (returns > 0).mean()

    out = {
        "total_return":   total,
        "ann_return":     ann_r,
        "ann_volatility": ann_v,
        "sharpe":         sharpe,
        "max_drawdown":   max_dd,
        "calmar":         calmar,
        "hit_ratio":      hit,
        "n_trades":       int((returns != 0).sum()),
    }

    if benchmark is not None:
        bm_total = (1 + benchmark).prod() - 1
        bm_ann_r = (1 + benchmark).prod() ** (ann / len(benchmark)) - 1
        out["bm_total_return"] = bm_total
        out["bm_ann_return"]   = bm_ann_r
        out["excess_return"]   = ann_r - bm_ann_r

    return out


def print_metrics(metrics: dict, label: str = "") -> None:
    if label:
        print(f"\n{'-'*40}")
        print(f"  {label}")
        print(f"{'-'*40}")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:<22} {v:+.4f}")
        else:
            print(f"  {k:<22} {v}")
