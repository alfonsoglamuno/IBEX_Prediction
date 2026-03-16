"""
Inference script — generates the current IBEX35 market signal.

Loads trained models (saved by train.py), runs on the latest feature row,
and writes results/latest_prediction.json for the dashboard.

Usage:
    python predict.py                         # uses best available model
    python predict.py --model xgboost
    python predict.py --shap                  # include SHAP breakdown
    python predict.py --no-multiasset
"""
from __future__ import annotations

import argparse
import json
from datetime import date

import numpy as np
import pandas as pd

from src.data.fetch import load_raw
from src.data.features import build_features, feature_cols
from src.data.multiasset import fetch_multiasset_features
from src.models.ensemble import best_available_model, load_model
from src.utils.config import get, root
from src.utils.logging import get_logger

log = get_logger(__name__)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",          default=None,
                   help="Model name (default: best available)")
    p.add_argument("--shap",           action="store_true")
    p.add_argument("--no-multiasset",  action="store_true")
    p.add_argument("--force-download", action="store_true")
    return p.parse_args()


def build_latest_features(force: bool = False, use_multiasset: bool = True) -> tuple[pd.DataFrame, list[str]]:
    df_raw = load_raw(force=force)
    df     = build_features(df_raw)

    if use_multiasset and get("multiasset.enabled", True):
        ma = fetch_multiasset_features(df.index)
        if not ma.empty:
            df = df.join(ma, how="left")

    feat_c = feature_cols(df)
    return df, feat_c


def predict_signal(
    model,
    model_name: str,
    df: pd.DataFrame,
    feat_c: list[str],
    include_shap: bool = False,
) -> dict:
    latest_row = df[feat_c].dropna().iloc[-1]
    latest_date = latest_row.name if hasattr(latest_row, "name") else df.index[-1]
    X_latest = latest_row.values.reshape(1, -1)

    prob_up = float(model.predict_proba(X_latest)[0, 1])

    # Signal: UP if prob_up > 0.55, DOWN if < 0.45, else NEUTRAL
    min_conf = get("backtest.min_confidence", 0.55)
    if prob_up >= min_conf:
        signal = "UP"
    elif prob_up <= (1 - min_conf):
        signal = "DOWN"
    else:
        signal = "NEUTRAL"

    confidence = "HIGH" if abs(prob_up - 0.5) > 0.15 else \
                 "MEDIUM" if abs(prob_up - 0.5) > 0.07 else "LOW"

    result = {
        "date":         str(latest_date.date() if hasattr(latest_date, "date") else latest_date),
        "model":        model_name,
        "prob_up":      round(prob_up, 4),
        "prob_down":    round(1 - prob_up, 4),
        "signal":       signal,
        "confidence":   confidence,
        "generated_at": date.today().isoformat(),
    }

    if include_shap:
        from src.explainability.shap_analysis import shap_for_latest
        shap_df = shap_for_latest(model, latest_row.values, feat_c)
        top_drivers = shap_df.head(10)[["feature", "feat_value", "shap_value"]].to_dict("records")
        result["top_drivers"] = top_drivers

    return result


def main():
    args = parse_args()
    use_multiasset = not args.no_multiasset

    df, feat_c = build_latest_features(
        force=args.force_download, use_multiasset=use_multiasset
    )
    log.info(f"Latest features date: {df.index[-1].date()} | {len(feat_c)} features")

    predictions = {}

    for target in ["target_dir_1d", "target_dir_5d"]:
        try:
            if args.model:
                model = load_model(args.model, target)
                model_name = args.model
            else:
                model, model_name = best_available_model(target)

            pred = predict_signal(model, model_name, df, feat_c,
                                  include_shap=args.shap)
            predictions[target] = pred
            log.info(
                f"{target}: signal={pred['signal']}  "
                f"P(up)={pred['prob_up']:.1%}  confidence={pred['confidence']}"
            )
        except FileNotFoundError as e:
            log.warning(str(e))
            predictions[target] = {"error": str(e)}

    out_path = root() / "results" / "latest_prediction.json"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(predictions, f, indent=2)
    log.info(f"Prediction saved -> {out_path}")

    # Print summary
    print("\n" + "=" * 50)
    print("  IBEX35 MARKET SIGNAL")
    print("=" * 50)
    for target, pred in predictions.items():
        if "error" in pred:
            print(f"  {target}: {pred['error']}")
        else:
            horizon = "1 dia" if "1d" in target else "5 dias"
            print(f"  {horizon:8s}  {pred['signal']:7s}  P(up)={pred['prob_up']:.1%}  [{pred['confidence']}]")
    print("=" * 50)


if __name__ == "__main__":
    main()
