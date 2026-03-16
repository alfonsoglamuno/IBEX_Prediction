"""
Baseline tabular models: Logistic Regression, Random Forest, XGBoost, LightGBM.
All share a common sklearn-compatible interface.
"""
from __future__ import annotations

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import xgboost as xgb
import lightgbm as lgb

from src.utils.config import get
from src.utils.logging import get_logger

log = get_logger(__name__)


def make_logistic(calibrate: bool = False) -> Pipeline:
    cfg = get("models.logistic_regression", {})
    clf = LogisticRegression(
        C=cfg.get("C", 0.1),
        max_iter=cfg.get("max_iter", 1000),
        class_weight=cfg.get("class_weight", "balanced"),
        solver="lbfgs",
        random_state=42,
    )
    pipe = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
    if calibrate:
        return CalibratedClassifierCV(pipe, method="isotonic", cv=3)
    return pipe


def make_random_forest(calibrate: bool = False) -> RandomForestClassifier:
    cfg = get("models.random_forest", {})
    clf = RandomForestClassifier(
        n_estimators=cfg.get("n_estimators", 300),
        max_depth=cfg.get("max_depth", 6),
        min_samples_leaf=cfg.get("min_samples_leaf", 20),
        class_weight=cfg.get("class_weight", "balanced"),
        n_jobs=cfg.get("n_jobs", -1),
        random_state=42,
    )
    if calibrate:
        return CalibratedClassifierCV(clf, method="isotonic", cv=3)
    return clf


def make_xgboost(scale_pos_weight: float = 1.0) -> xgb.XGBClassifier:
    cfg = get("models.xgboost", {})
    return xgb.XGBClassifier(
        n_estimators=cfg.get("n_estimators", 500),
        max_depth=cfg.get("max_depth", 4),
        learning_rate=cfg.get("learning_rate", 0.01),
        subsample=cfg.get("subsample", 0.8),
        colsample_bytree=cfg.get("colsample_bytree", 0.8),
        scale_pos_weight=scale_pos_weight,
        eval_metric=cfg.get("eval_metric", "logloss"),
        early_stopping_rounds=cfg.get("early_stopping_rounds", 30),
        # NOTE: use_label_encoder removed in XGBoost 1.6+
        random_state=42,
        verbosity=0,
    )


def make_lgbm(scale_pos_weight: float = 1.0) -> lgb.LGBMClassifier:
    cfg = get("models.lgbm", {})
    return lgb.LGBMClassifier(
        n_estimators=cfg.get("n_estimators", 500),
        max_depth=cfg.get("max_depth", 4),
        learning_rate=cfg.get("learning_rate", 0.01),
        subsample=cfg.get("subsample", 0.8),
        colsample_bytree=cfg.get("colsample_bytree", 0.8),
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        verbose=-1,
    )


REGISTRY: dict[str, callable] = {
    "logistic":      make_logistic,
    "random_forest": make_random_forest,
    "xgboost":       make_xgboost,
    "lgbm":          make_lgbm,
}


def get_model(name: str, **kwargs):
    if name not in REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Available: {list(REGISTRY)}")
    return REGISTRY[name](**kwargs)
