"""
SHAP-based explainability for IBEX35 forecasting models.

Uses the modern shap.Explainer API (compatible with shap>=0.44, XGBoost>=2).

Supports:
  - Tree models  : XGBoost, LightGBM, RandomForest
  - Linear models: LogisticRegression (via pipeline)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.logging import get_logger

log = get_logger(__name__)


def _tree_types():
    import xgboost as xgb
    import lightgbm as lgb
    from sklearn.ensemble import RandomForestClassifier
    return (xgb.XGBClassifier, lgb.LGBMClassifier, RandomForestClassifier)


def _unwrap_pipeline(model):
    """Return (scaler_or_None, final_estimator) from any sklearn wrapper."""
    from sklearn.pipeline import Pipeline
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.ensemble import StackingClassifier

    # CalibratedClassifierCV: .estimator is UNFITTED; use calibrated_classifiers_ for the fitted one
    if isinstance(model, CalibratedClassifierCV):
        if hasattr(model, "calibrated_classifiers_") and model.calibrated_classifiers_:
            model = model.calibrated_classifiers_[0].estimator
        else:
            model = model.estimator
    if isinstance(model, Pipeline):
        scaler = dict(model.steps).get("scaler", None)
        clf    = model.steps[-1][1]
        # If the final step is a stacking ensemble, pick the best single model
        if isinstance(clf, StackingClassifier):
            # Prefer first tree estimator inside the stack
            for _, est in clf.estimators:
                if isinstance(est, _tree_types()):
                    return scaler, est
        return scaler, clf
    return None, model


def _extract_shap_values(raw, n_features: int) -> np.ndarray:
    """
    Normalise SHAP output to shape (n_samples, n_features) for class-1.
    Handles: plain ndarray, list-of-arrays, shap.Explanation objects.
    """
    import shap

    # Modern API returns Explanation object
    if isinstance(raw, shap.Explanation):
        vals = raw.values
        if vals.ndim == 3:          # (n, features, classes)
            return vals[:, :, 1].astype(float)
        if vals.ndim == 2:
            return vals.astype(float)
        return vals.reshape(-1, n_features).astype(float)

    # Legacy list-of-arrays  [class0_array, class1_array]
    if isinstance(raw, list):
        arr = np.array(raw[1] if len(raw) == 2 else raw[0], dtype=float)
    else:
        arr = np.array(raw, dtype=float)

    if arr.ndim == 3:               # (n, features, classes)
        return arr[:, :, 1]
    return arr


def _align_X(X: np.ndarray, feature_names: list[str], clf) -> tuple[np.ndarray, list[str]]:
    """If model was trained on fewer features, slice X to match."""
    n_expected = getattr(clf, "n_features_in_", None)
    if n_expected is not None and X.shape[1] != n_expected:
        X = X[:, :n_expected]
        feature_names = feature_names[:n_expected]
    return X, feature_names


def compute_shap(
    model,
    X: np.ndarray,
    feature_names: list[str],
    max_samples: int = 500,
) -> tuple[np.ndarray, list[str]]:
    """
    Compute SHAP values for class 1 (up).
    Returns (shap_values array shape (n, n_features), feature_names).
    """
    try:
        import shap
    except ImportError:
        raise ImportError("Install shap:  pip install shap")

    if len(X) > max_samples:
        idx = np.random.choice(len(X), max_samples, replace=False)
        idx.sort()
        X = X[idx]

    scaler, clf = _unwrap_pipeline(model)
    X_sc = scaler.transform(X) if scaler is not None else X
    X_in, feature_names = _align_X(X_sc, feature_names, clf)

    try:
        if isinstance(clf, _tree_types()):
            explainer = shap.TreeExplainer(clf)
            raw       = explainer(X_in)
        else:
            background = shap.sample(X_in, min(100, len(X_in)))
            explainer  = shap.LinearExplainer(clf, background)
            raw        = explainer(X_in)

        vals = _extract_shap_values(raw, len(feature_names))
        if vals.shape[1] != len(feature_names):
            raise ValueError(f"SHAP shape {vals.shape} != n_features {len(feature_names)}")

        log.info(f"SHAP computed: {vals.shape}")
        return vals, feature_names

    except Exception as e:
        log.warning(f"SHAP computation failed ({e}), returning zeros")
        return np.zeros((len(X_in), len(feature_names))), feature_names


def shap_summary(
    shap_values: np.ndarray,
    feature_names: list[str],
    top_n: int = 20,
) -> pd.DataFrame:
    """Mean |SHAP| per feature, sorted descending."""
    importance = np.abs(shap_values).mean(axis=0)
    return (
        pd.DataFrame({"feature": feature_names, "mean_shap": importance})
        .sort_values("mean_shap", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )


def shap_for_latest(
    model,
    X_latest: np.ndarray,
    feature_names: list[str],
) -> pd.DataFrame:
    """
    SHAP values for a single observation (latest row).
    Returns DataFrame with columns: feature, feat_value, shap_value, abs_shap.
    """
    try:
        import shap
    except ImportError:
        raise ImportError("Install shap:  pip install shap")

    scaler, clf = _unwrap_pipeline(model)
    x    = X_latest.reshape(1, -1)
    x_sc = scaler.transform(x) if scaler is not None else x
    X_in, feature_names = _align_X(x_sc, feature_names, clf)
    feat_values = X_latest[:len(feature_names)]

    try:
        if isinstance(clf, _tree_types()):
            explainer = shap.TreeExplainer(clf)
            raw       = explainer(X_in)
        else:
            background = shap.sample(X_in, 1)
            explainer  = shap.LinearExplainer(clf, background)
            raw        = explainer(X_in)

        vals = _extract_shap_values(raw, len(feature_names))
        row  = vals[0].astype(float)

    except Exception as e:
        log.warning(f"SHAP latest failed: {e}")
        row = np.zeros(len(feature_names), dtype=float)

    df = pd.DataFrame({
        "feature":    feature_names,
        "feat_value": feat_values,
        "shap_value": row,
        "abs_shap":   np.abs(row),
    })
    return df.sort_values("abs_shap", ascending=False).reset_index(drop=True)
