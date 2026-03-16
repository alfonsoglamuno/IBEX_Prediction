"""
SHAP-based explainability for IBEX35 forecasting models.

Supports:
  - TreeExplainer  → XGBoost, LightGBM, RandomForest
  - LinearExplainer → LogisticRegression (via pipeline)
  - Aggregated feature importance across folds / samples

Usage:
    from src.explainability.shap_analysis import compute_shap, shap_summary
    shap_vals, feat_names = compute_shap(model, X, feature_names)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.logging import get_logger

log = get_logger(__name__)

_TREE_TYPES = None   # populated lazily to avoid import at module level


def _tree_types():
    global _TREE_TYPES
    if _TREE_TYPES is None:
        import xgboost as xgb
        import lightgbm as lgb
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
        _TREE_TYPES = (xgb.XGBClassifier, lgb.LGBMClassifier,
                       RandomForestClassifier, GradientBoostingClassifier)
    return _TREE_TYPES


def _unwrap_pipeline(model):
    """Extract the final estimator from sklearn Pipeline if needed."""
    from sklearn.pipeline import Pipeline
    from sklearn.calibration import CalibratedClassifierCV
    if isinstance(model, CalibratedClassifierCV):
        model = model.estimator
    if isinstance(model, Pipeline):
        # Return (scaler_or_None, final_estimator)
        steps = dict(model.steps)
        clf = model.steps[-1][1]
        scaler = steps.get("scaler", None)
        return scaler, clf
    return None, model


def compute_shap(
    model,
    X: np.ndarray,
    feature_names: list[str],
    max_samples: int = 500,
) -> tuple[np.ndarray, list[str]]:
    """
    Compute SHAP values for class 1 (up) predictions.

    Returns
    -------
    shap_values : np.ndarray, shape (n_samples, n_features)
    feature_names : list[str]
    """
    try:
        import shap
    except ImportError:
        raise ImportError("Install shap:  pip install shap")

    # Subsample for speed
    if len(X) > max_samples:
        idx = np.random.choice(len(X), max_samples, replace=False)
        idx.sort()
        X = X[idx]

    scaler, clf = _unwrap_pipeline(model)
    X_in = scaler.transform(X) if scaler is not None else X

    try:
        if isinstance(clf, _tree_types()):
            explainer  = shap.TreeExplainer(clf)
            shap_raw   = explainer.shap_values(X_in)
            # For binary classification trees: shap_values is list [class0, class1]
            if isinstance(shap_raw, list):
                vals = shap_raw[1]
            else:
                vals = shap_raw
        else:
            # Linear / fallback
            background = shap.sample(X_in, min(100, len(X_in)))
            explainer  = shap.LinearExplainer(clf, background)
            vals       = explainer.shap_values(X_in)
            if isinstance(vals, list):
                vals = vals[1]

        log.info(f"SHAP computed: {vals.shape}")
        return vals, feature_names

    except Exception as e:
        log.warning(f"SHAP computation failed ({e}), returning zeros")
        return np.zeros((len(X), len(feature_names))), feature_names


def shap_summary(
    shap_values: np.ndarray,
    feature_names: list[str],
    top_n: int = 20,
) -> pd.DataFrame:
    """
    Returns a DataFrame of mean |SHAP| per feature, sorted descending.
    """
    importance = np.abs(shap_values).mean(axis=0)
    df = pd.DataFrame({
        "feature":    feature_names,
        "mean_shap":  importance,
    }).sort_values("mean_shap", ascending=False).head(top_n).reset_index(drop=True)
    return df


def shap_for_latest(
    model,
    X_latest: np.ndarray,
    feature_names: list[str],
) -> pd.DataFrame:
    """
    Compute SHAP for a single row (latest observation).
    Returns a DataFrame with feature, value, shap_value, sorted by |shap|.
    """
    try:
        import shap
    except ImportError:
        raise ImportError("Install shap:  pip install shap")

    scaler, clf = _unwrap_pipeline(model)
    X_in = scaler.transform(X_latest.reshape(1, -1)) if scaler is not None \
           else X_latest.reshape(1, -1)

    try:
        if isinstance(clf, _tree_types()):
            explainer = shap.TreeExplainer(clf)
            raw       = explainer.shap_values(X_in)
            vals      = raw[1][0] if isinstance(raw, list) else raw[0]
        else:
            background = shap.sample(X_in, 1)
            explainer  = shap.LinearExplainer(clf, background)
            raw        = explainer.shap_values(X_in)
            vals       = raw[1][0] if isinstance(raw, list) else raw[0]

        df = pd.DataFrame({
            "feature":     feature_names,
            "feat_value":  X_latest,
            "shap_value":  vals,
        })
        df["abs_shap"] = df["shap_value"].abs()
        return df.sort_values("abs_shap", ascending=False).reset_index(drop=True)

    except Exception as e:
        log.warning(f"SHAP latest failed: {e}")
        return pd.DataFrame({
            "feature":    feature_names,
            "feat_value": X_latest,
            "shap_value": np.zeros(len(feature_names)),
            "abs_shap":   np.zeros(len(feature_names)),
        })
