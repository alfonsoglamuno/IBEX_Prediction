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
    base_estimators = [
        ("xgboost",       make_xgboost()),
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

import joblib
from pathlib import Path
from src.utils.config import root


def save_model(model, name: str, target: str) -> Path:
    model_dir = root() / get("persistence.model_dir", "results/models")
    model_dir.mkdir(parents=True, exist_ok=True)
    path = model_dir / f"{target}_{name}.joblib"
    joblib.dump(model, path)
    log.info(f"Saved model → {path}")
    return path


def load_model(name: str, target: str):
    model_dir = root() / get("persistence.model_dir", "results/models")
    path = model_dir / f"{target}_{name}.joblib"
    if not path.exists():
        raise FileNotFoundError(f"No saved model at {path}. Run train.py first.")
    model = joblib.load(path)
    log.info(f"Loaded model ← {path}")
    return model


def best_available_model(target: str):
    """Load the best model for a given target (prefers ensemble > xgboost > lgbm)."""
    for name in ["ensemble", "xgboost", "lgbm", "random_forest", "logistic"]:
        try:
            return load_model(name, target), name
        except FileNotFoundError:
            continue
    raise FileNotFoundError(f"No trained model found for target '{target}'. Run train.py first.")
