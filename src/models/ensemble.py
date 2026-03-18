"""
Stacking ensemble for IBEX35 directional prediction.

Uses XGBoost + LightGBM + RandomForest as base learners,
LogisticRegression as meta-learner (with cross-validation stacking).
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV

from src.models.baseline import make_xgboost, make_lgbm, make_random_forest
from src.utils.config import get
from src.utils.logging import get_logger

log = get_logger(__name__)


def make_ensemble() -> Pipeline:
    """
    Returns a sklearn Pipeline:
      - StandardScaler (for meta-learner)
      - StackingClassifier with XGB + LGBM + RF base models, LogReg meta

    Base models do NOT use the scaler (tree models don't need it).
    The stacking CV uses 5-fold time-aware splitting handled by sklearn's
    default StratifiedKFold (acceptable for stacking meta-features).
    """
    # StackingClassifier trains base models via internal CV without eval_set,
    # so early_stopping_rounds must be disabled for XGBoost and LightGBM here.
    xgb_base = make_xgboost()
    xgb_base.set_params(early_stopping_rounds=None)
    base_estimators = [
        ("xgboost",       xgb_base),
        ("lgbm",          make_lgbm()),
        ("random_forest", make_random_forest()),
    ]
    meta = LogisticRegression(
        C=0.1, max_iter=1000, solver="lbfgs",
        class_weight="balanced", random_state=42,
    )
    stacking = StackingClassifier(
        estimators=base_estimators,
        final_estimator=meta,
        stack_method="predict_proba",
        cv=5,
        n_jobs=-1,
        passthrough=False,    # only pass stacked probabilities to meta
    )
    return Pipeline([
        ("scaler", StandardScaler()),
        ("ensemble", stacking),
    ])


# ── Model persistence ─────────────────────────────────────────────────────────

import json
import joblib
from pathlib import Path
from src.utils.config import root


def save_model(model, name: str, target: str,
               feature_names: list[str] | None = None) -> Path:
    model_dir = root() / get("persistence.model_dir", "results/models")
    model_dir.mkdir(parents=True, exist_ok=True)
    path = model_dir / f"{target}_{name}.joblib"
    joblib.dump({"model": model, "feature_names": feature_names}, path)
    log.info(f"Saved model -> {path}")
    return path


def load_model(name: str, target: str):
    """Returns the model object (feature_names accessible via load_model_bundle)."""
    bundle = _load_bundle(name, target)
    return bundle["model"]


def load_model_bundle(name: str, target: str) -> dict:
    """Returns dict with keys 'model' and 'feature_names'."""
    return _load_bundle(name, target)


def _load_bundle(name: str, target: str) -> dict:
    model_dir = root() / get("persistence.model_dir", "results/models")
    path      = model_dir / f"{target}_{name}.joblib"
    if not path.exists():
        raise FileNotFoundError(f"No saved model at {path}. Run train.py first.")
    raw = joblib.load(path)
    # Handle both old format (bare model) and new format (dict with feature_names)
    if isinstance(raw, dict) and "model" in raw:
        log.info(f"Loaded model <- {path}")
        return raw
    return {"model": raw, "feature_names": None}


def best_available_model(target: str):
    """
    Load the walk-forward champion model for target.

    Selection priority:
      1. champion.json (written by train.py after model comparison)
      2. Fallback preference order: ensemble > xgboost > lgbm > random_forest > logistic
    """
    # Check champion registry first
    champion_path = root() / "results" / "champion.json"
    if champion_path.exists():
        try:
            with open(champion_path) as f:
                champions = json.load(f)
            if target in champions:
                name = champions[target]["model"]
                log.info(f"Champion model for {target}: {name} "
                         f"(AUC={champions[target].get('roc_auc', '?'):.4f})")
                return load_model(name, target), name
        except Exception as exc:
            log.debug(f"Could not read champion.json: {exc}")

    # Fallback: first model file that exists
    for name in ["ensemble", "xgboost", "lgbm", "random_forest", "logistic"]:
        try:
            return load_model(name, target), name
        except FileNotFoundError:
            continue
    raise FileNotFoundError(f"No trained model found for target '{target}'. Run train.py first.")
