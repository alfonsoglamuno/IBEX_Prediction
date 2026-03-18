"""
Main training script — V1 pipeline.

Usage:
    python train.py                          # all baselines, target_dir_1d
    python train.py --target target_dir_5d
    python train.py --model xgboost
    python train.py --model ensemble
    python train.py --all-targets            # train all targets sequentially
    python train.py --force-download
    python train.py --no-multiasset
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.fetch import load_raw
from src.data.features import build_features, feature_cols
from src.data.multiasset import fetch_multiasset_features
from src.evaluation.backtest import walk_forward_cv
from src.evaluation.metrics import print_metrics
from src.models.baseline import get_model, REGISTRY
from src.models.ensemble import make_ensemble, save_model
from src.utils.config import get, root
from src.utils.logging import get_logger

log = get_logger(__name__)

# target_ret_5d excluded: registry contains classifiers only; add XGBRegressor to support it
ALL_TARGETS = ["target_dir_1d", "target_dir_5d"]
TASK_MAP    = {
    "target_dir_1d":    "classification",
    "target_dir_5d":    "classification",
    "target_ret_5d":    "regression",
    "target_vol_regime":"classification",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--target",       default="target_dir_1d",
                   choices=list(TASK_MAP))
    p.add_argument("--all-targets",  action="store_true",
                   help="Train all classification + regression targets")
    p.add_argument("--model",        default="all",
                   choices=list(REGISTRY) + ["all", "ensemble", "lstm"])
    p.add_argument("--force-download", action="store_true")
    p.add_argument("--no-multiasset",  action="store_true")
    return p.parse_args()


def build_full_feature_matrix(df_raw: pd.DataFrame,
                               use_multiasset: bool = True) -> pd.DataFrame:
    """Build feature matrix with optional multi-asset and sentiment features.

    Sentiment is handled inside build_features() when include_sentiment: true.
    Multi-asset features are appended here separately.
    """
    df = build_features(df_raw)

    if use_multiasset and get("multiasset.enabled", True):
        ma = fetch_multiasset_features(df.index)
        if not ma.empty:
            df = df.join(ma, how="left")
            log.info(f"Added {ma.shape[1]} multi-asset features")

    return df


def train_target(df: pd.DataFrame, target: str, models_to_run: list[str]) -> dict:
    """Train all models for one target, return summary dict."""
    df_t = df.dropna(subset=[target])
    task = TASK_MAP.get(target, "classification")
    log.info(f"\n{'='*60}\nTarget: {target} | Task: {task} | Rows: {len(df_t)}\n{'='*60}")

    results    = {}
    all_preds  = {}

    for model_name in models_to_run:
        if model_name == "lstm":
            log.info("LSTM: run lstm_train.py separately")
            continue

        log.info(f"  -> Training {model_name}")

        if model_name == "ensemble":
            make_fn = make_ensemble
        else:
            # Capture model_name by value via default arg
            def make_fn(name=model_name):
                return get_model(name)

        result = walk_forward_cv(
            df_feat   = df_t,
            target    = target,
            make_model= make_fn,
            task      = task,
            verbose   = True,
        )

        results[model_name] = result.summary
        print_metrics(result.summary, label=f"{model_name} | {target}")

        # ── Save predictions ──────────────────────────────────────────────────
        if result.all_predictions is not None:
            pred_dir = root() / get("persistence.predictions_dir", "results")
            pred_dir.mkdir(exist_ok=True)
            pred_path = pred_dir / f"{target}_{model_name}_predictions.parquet"
            result.all_predictions.to_parquet(pred_path)
            all_preds[model_name] = result.all_predictions

            # ── Compute percentile signal thresholds from OOS probs ───────────
            if "y_prob" in result.all_predictions.columns:
                _save_signal_thresholds(target, model_name,
                                        result.all_predictions["y_prob"])

        # ── Retrain on full data and save model ───────────────────────────────
        _retrain_and_save(df_t, target, task, model_name, make_fn)

    # ── Save JSON summary ─────────────────────────────────────────────────────
    out_dir  = root() / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"{target}_summary.json"
    with open(out_path, "w") as f:
        json.dump(
            {k: {kk: round(vv, 6) for kk, vv in v.items() if isinstance(vv, float)}
             for k, v in results.items()},
            f, indent=2,
        )
    log.info(f"Summary saved -> {out_path}")
    return results


def _retrain_and_save(df: pd.DataFrame, target: str, task: str,
                      model_name: str, make_fn) -> None:
    """Retrain on 80% of data and save the final model."""
    feat_c = feature_cols(df)
    X = np.nan_to_num(df[feat_c].values, nan=0.0)
    y = df[target].values

    split_idx = int(len(X) * 0.8)
    X_tr, y_tr = X[:split_idx], y[:split_idx]
    X_va, y_va = X[split_idx:], y[split_idx:]

    model = make_fn()
    try:
        if task == "classification":
            model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        else:
            model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    except (TypeError, ValueError):
        model.fit(X_tr, y_tr)

    save_model(model, model_name, target, feature_names=feat_c)


def _save_signal_thresholds(target: str, model_name: str, probs: pd.Series) -> None:
    """
    Compute percentile-based UP/DOWN thresholds from OOS walk-forward probabilities
    and persist to results/signal_thresholds.json.

    Thresholds are model-specific: a calibrated model (std≈0.04) has tight thresholds
    while an uncalibrated model (std≈0.15) has wide ones. Both correctly represent
    "top quartile of this model's own conviction distribution".
    """
    pct         = get("signal.percentile", 0.75)
    probs_clean = probs.dropna()
    if len(probs_clean) < 50:
        log.warning(f"Too few OOS samples ({len(probs_clean)}) to compute thresholds for {target}_{model_name}")
        return

    thresholds_path = root() / "results" / "signal_thresholds.json"
    existing: dict = {}
    if thresholds_path.exists():
        with open(thresholds_path) as _f:
            existing = json.load(_f)

    key = f"{target}_{model_name}"
    existing[key] = {
        "up_threshold":   round(float(np.percentile(probs_clean, pct * 100)),       6),
        "down_threshold": round(float(np.percentile(probs_clean, (1 - pct) * 100)), 6),
        "p90":            round(float(np.percentile(probs_clean, 90)), 6),
        "p10":            round(float(np.percentile(probs_clean, 10)), 6),
        "p50":            round(float(np.percentile(probs_clean, 50)), 6),
        "n_samples":      int(len(probs_clean)),
        "signal_pct":     pct,
        "mode":           "percentile",
    }
    with open(thresholds_path, "w") as _f:
        json.dump(existing, _f, indent=2)

    t = existing[key]
    log.info(f"Signal thresholds [{key}]: "
             f"UP≥{t['up_threshold']:.4f}  DOWN≤{t['down_threshold']:.4f}  "
             f"(top/bottom {(1-pct):.0%} of {t['n_samples']} OOS samples)")


def select_champion(all_results: dict[str, dict]) -> dict:
    """
    Given {target: {model_name: summary_metrics}}, select the best model per target
    and write results/champion.json.

    Selection metric: roc_auc (primary) + bt_sharpe (tiebreaker).
    For regression targets: uses ic (information coefficient).
    """
    champions: dict = {}
    for target, model_results in all_results.items():
        task = TASK_MAP.get(target, "classification")
        best_name, best_score, best_metrics = None, -999.0, {}

        for model_name, metrics in model_results.items():
            if not metrics:
                continue
            if task == "classification":
                # Combined score: 60% AUC + 40% Sharpe (normalised to [0,1] range)
                auc    = metrics.get("roc_auc", 0.5)
                sharpe = metrics.get("bt_sharpe", 0.0)
                score  = 0.6 * auc + 0.4 * max(0.0, min(sharpe / 3.0, 1.0))
            else:
                score = metrics.get("ic", 0.0)

            if score > best_score:
                best_score, best_name, best_metrics = score, model_name, metrics

        if best_name:
            champions[target] = {
                "model":      best_name,
                "score":      round(best_score, 6),
                "roc_auc":    round(best_metrics.get("roc_auc", 0.0), 6),
                "bt_sharpe":  round(best_metrics.get("bt_sharpe", 0.0), 4),
                "selected_at": str(pd.Timestamp.now().date()),
            }
            log.info(f"Champion [{target}]: {best_name:15s}  "
                     f"AUC={champions[target]['roc_auc']:.4f}  "
                     f"Sharpe={champions[target]['bt_sharpe']:.3f}")

    champion_path = root() / "results" / "champion.json"
    with open(champion_path, "w") as f:
        json.dump(champions, f, indent=2)
    log.info(f"Champion models saved -> {champion_path}")
    return champions


def main():
    args         = parse_args()
    use_multiasset = not args.no_multiasset

    # ── 1. Data ───────────────────────────────────────────────────────────────
    df_raw = load_raw(force=args.force_download)
    log.info(f"Raw: {df_raw.index[0].date()} to {df_raw.index[-1].date()} "
             f"({len(df_raw)} rows)")

    # ── 2. Features ───────────────────────────────────────────────────────────
    df = build_full_feature_matrix(df_raw, use_multiasset=use_multiasset)
    log.info(f"Feature matrix: {df.shape[1]} cols | {len(df)} rows")

    # ── 3. Models ─────────────────────────────────────────────────────────────
    if args.model == "all":
        models_to_run = list(REGISTRY) + ["ensemble"]
    else:
        models_to_run = [args.model]

    # ── 4. Train ──────────────────────────────────────────────────────────────
    targets = ALL_TARGETS if args.all_targets else [args.target]

    all_results: dict[str, dict] = {}
    for target in targets:
        all_results[target] = train_target(df, target, models_to_run)

    # ── 5. Auto-select champion model per target ───────────────────────────────
    if len(all_results) > 0 and len(models_to_run) > 1:
        champions = select_champion(all_results)
        print("\n" + "=" * 55)
        print("  MODEL SELECTION RESULTS")
        print("=" * 55)
        for target, ch in champions.items():
            print(f"  {target:25s}  champion={ch['model']:15s}  "
                  f"AUC={ch['roc_auc']:.4f}  Sharpe={ch['bt_sharpe']:.3f}")
        print("=" * 55)
        print("\nRun: python predict.py --shap")
        print("The champion model will be used automatically.\n")


if __name__ == "__main__":
    main()
