"""
Main training script.

Usage:
    python train.py                          # all models, direction_1d target
    python train.py --target target_dir_5d
    python train.py --model xgboost
    python train.py --force-download
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.data.fetch import load_raw
from src.data.features import build_features, feature_cols
from src.evaluation.backtest import walk_forward_cv
from src.evaluation.metrics import print_metrics
from src.models.baseline import get_model, REGISTRY
from src.utils.config import get, root
from src.utils.logging import get_logger

log = get_logger(__name__)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--target",   default="target_dir_1d",
                   choices=["target_dir_1d", "target_dir_5d", "target_ret_5d", "target_vol_regime"])
    p.add_argument("--model",    default="all",
                   choices=list(REGISTRY) + ["all", "lstm"])
    p.add_argument("--force-download", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    log.info(f"Target: {args.target} | Model: {args.model}")

    # 1. Data
    df_raw = load_raw(force=args.force_download)
    log.info(f"Raw data: {df_raw.index[0].date()} → {df_raw.index[-1].date()} ({len(df_raw)} rows)")

    # 2. Features
    df = build_features(df_raw)
    feat_cols = feature_cols(df)
    log.info(f"Features: {len(feat_cols)} columns | Rows after NaN trim: {len(df)}")

    # 3. Drop rows where target is NaN (forward-looking targets are NaN at the end)
    df = df.dropna(subset=[args.target])
    log.info(f"Rows after target NaN drop: {len(df)}")

    task = "regression" if args.target == "target_ret_5d" else "classification"

    # 4. Train / evaluate
    models_to_run = list(REGISTRY) if args.model == "all" else [args.model]

    results = {}
    for model_name in models_to_run:
        if model_name == "lstm":
            continue  # LSTM uses separate trainer (lstm_train.py)
        log.info(f"\n{'='*50}\nModel: {model_name}\n{'='*50}")

        def make_model(name=model_name):
            m = get_model(name)
            return m

        result = walk_forward_cv(
            df_feat=df,
            target=args.target,
            make_model=make_model,
            task=task,
            verbose=True,
        )
        results[model_name] = result.summary
        print_metrics(result.summary, label=f"{model_name} | {args.target}")

    # 5. Save summary
    out_dir = root() / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"{args.target}_summary.json"
    with open(out_path, "w") as f:
        json.dump(
            {k: {kk: round(vv, 6) for kk, vv in v.items() if isinstance(vv, float)}
             for k, v in results.items()},
            f, indent=2
        )
    log.info(f"Results saved → {out_path}")


if __name__ == "__main__":
    main()
