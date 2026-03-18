"""
Daily sentiment feature construction for IBEX35 forecasting.

Three-layer architecture (Tetlock 2007; Loughran & McDonald 2011; Baker & Wurgler 2007):

  Layer 1 — Direct IBEX sentiment (articles explicitly mentioning IBEX35 / bolsa)
  Layer 2 — Constituent roll-up  (company-level articles weighted by index weight)
  Layer 3 — Macro EU/Spain       (ECB, rates, eurozone articles)

Composite index (hybrid formula):
  IndexSent_t = α·DirectIBEXSent_t + β·ConstituentRollup_t + γ·MacroSpainEU_t
  Default: α=0.50, β=0.30, γ=0.20

Per-article weight (Eq. 1):
  w_i = w_rel · w_src · w_time · w_novelty

Feature block (all lagged by 1 full trading day — no lookahead):
  sent_direct_1d    — 1d direct-IBEX weighted sentiment
  sent_direct_3d    — 3d EMA of direct-IBEX sentiment
  sent_direct_5d    — 5d EMA of direct-IBEX sentiment
  sent_macro_1d     — 1d macro layer sentiment
  sent_const_rollup_1d — 1d constituent roll-up sentiment
  sent_composite_1d — 1d hybrid composite (α·direct + β·rollup + γ·macro)
  sent_neg_share_1d — share of negative articles among relevant articles
  sent_dispersion_1d — std dev of scores (uncertainty signal)
  sent_count_1d     — article count (normalised log)
  sent_shock_count_1d — count of |score| > 0.5 articles
  sent_zscore_20d   — z-score of composite vs 20d rolling mean
  sent_change_1d    — 1d change in composite score
  sent_abs_mean_1d  — mean |score| (absolute sentiment intensity)

Literature references:
  - Tetlock (2007): media pessimism predicts next-day Dow Jones returns
  - Tetlock, Saar-Tsechansky & Macskassy (2008): negative words predict earnings
  - Loughran & McDonald (2011): finance-specific lexicon; L&M wordlists
  - Baker & Wurgler (2007): investor sentiment and cross-section of stock returns
  - Groß-Klußmann & Hautsch (2011): news intensity and limit order book
  - Da, Engelberg & Gao (2011): in-sample news decay half-life ~1 day
  - Garcia (2013): newspaper tone predicts stock returns (especially during recessions)
"""
from __future__ import annotations

from datetime import timezone
from typing import Optional

import numpy as np
import pandas as pd

from src.data.news_loader import load_all
from src.nlp.sentiment_scoring import score_batch, ScoredArticle
from src.utils.config import get
from src.utils.logging import get_logger

log = get_logger(__name__)

# Composite layer weights (α, β, γ) — can be overridden in config.yaml
_ALPHA = 0.50   # direct IBEX sentiment
_BETA  = 0.30   # constituent roll-up
_GAMMA = 0.20   # macro EU/Spain


def _safe_weighted_mean(scores: list[float], weights: list[float]) -> float:
    if not scores:
        return 0.0
    w_arr = np.array(weights, dtype=float)
    s_arr = np.array(scores,  dtype=float)
    total = w_arr.sum()
    return float((s_arr * w_arr).sum() / total) if total > 0 else 0.0


def _aggregate_day(articles: list[ScoredArticle]) -> dict:
    """Compute all sentiment signals for a single trading day's article set."""
    direct  = [a for a in articles if a.target_type == "index"]
    const   = [a for a in articles if a.target_type == "constituent"]
    macro   = [a for a in articles if a.target_type == "macro"]
    relevant = direct + const + macro

    def wmean(group: list[ScoredArticle]) -> float:
        return _safe_weighted_mean(
            [a.sentiment.score for a in group],
            [a.combined_weight for a in group],
        )

    direct_sent = wmean(direct)
    macro_sent  = wmean(macro)

    # Constituent roll-up: weight by index constituent weight × article weight
    if const:
        rollup_scores  = [a.sentiment.score for a in const]
        rollup_weights = [a.combined_weight * a.constituent_weight for a in const]
        const_rollup = _safe_weighted_mean(rollup_scores, rollup_weights)
    else:
        const_rollup = 0.0

    composite = (_ALPHA * direct_sent + _BETA * const_rollup + _GAMMA * macro_sent)

    all_scores = [a.sentiment.score for a in relevant]
    neg_share  = sum(1 for s in all_scores if s < -0.05) / max(1, len(all_scores))
    dispersion = float(np.std(all_scores)) if len(all_scores) > 1 else 0.0
    shock_count = sum(1 for s in all_scores if abs(s) > 0.5)
    count_log   = float(np.log1p(len(relevant)))

    return {
        "sent_direct_1d":      direct_sent,
        "sent_macro_1d":       macro_sent,
        "sent_const_rollup_1d": const_rollup,
        "sent_composite_1d":   composite,
        "sent_neg_share_1d":   neg_share,
        "sent_dispersion_1d":  dispersion,
        "sent_count_1d":       count_log,
        "sent_shock_count_1d": float(shock_count),
        "sent_abs_mean_1d":    float(np.mean(np.abs(all_scores))) if all_scores else 0.0,
    }


def build_sentiment_features(
    ibex_index: pd.DatetimeIndex,
    days_back: int = 365,
    lang: str = "es",
    backend: str = "auto",
) -> pd.DataFrame:
    """
    Build the full sentiment feature block aligned to ibex_index.

    All features are lagged by 1 trading day (shift(1)) — no lookahead.
    Missing days are forward-filled (news effect persists until new signal).

    Returns DataFrame with 13 columns (see module docstring).
    """
    alpha = get("sentiment.alpha", _ALPHA)
    beta  = get("sentiment.beta",  _BETA)
    gamma = get("sentiment.gamma", _GAMMA)

    log.info("Loading news articles...")
    df_news = load_all(days_back=days_back)

    if df_news.empty:
        log.warning("No news data available — returning zero sentiment features")
        return _zero_features(ibex_index)

    # Score articles
    articles_raw = df_news.to_dict("records")
    scored: list[ScoredArticle] = score_batch(
        articles_raw, lang=lang, backend=backend
    )

    # Group scored articles by trading day (use date in UTC)
    day_map: dict[str, list[ScoredArticle]] = {}
    for art in scored:
        d = art.published.astimezone(timezone.utc).date().isoformat()
        day_map.setdefault(d, []).append(art)

    # Aggregate per day
    rows = []
    for date_str, arts in day_map.items():
        row = _aggregate_day(arts)
        row["date"] = pd.Timestamp(date_str)
        rows.append(row)

    if not rows:
        return _zero_features(ibex_index)

    daily = pd.DataFrame(rows).set_index("date")
    daily.index = pd.to_datetime(daily.index, utc=True).tz_localize(None)
    daily = daily.sort_index()

    # Align to ibex_index (trading days only, fill gaps forward)
    out = pd.DataFrame(index=ibex_index)
    for col in daily.columns:
        out[col] = daily[col].reindex(ibex_index, method="ffill")

    # Derived features
    composite = out["sent_composite_1d"].fillna(0.0)
    out["sent_direct_3d"] = composite.ewm(span=3, min_periods=1).mean()
    out["sent_direct_5d"] = composite.ewm(span=5, min_periods=1).mean()

    roll20 = composite.rolling(20, min_periods=5)
    out["sent_zscore_20d"] = (composite - roll20.mean()) / (roll20.std().replace(0, np.nan))
    out["sent_zscore_20d"] = out["sent_zscore_20d"].fillna(0.0)

    out["sent_change_1d"] = composite.diff(1)

    # All features lagged by 1 trading day
    lag_cols = [c for c in out.columns]
    out[lag_cols] = out[lag_cols].shift(1)

    # Fill remaining NaN with 0 (no sentiment = neutral)
    out = out.fillna(0.0)

    log.info(f"Sentiment features built: {out.shape[1]} features, "
             f"{(out['sent_composite_1d'] != 0).sum()} non-zero days")
    return out


def _zero_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    cols = [
        "sent_direct_1d", "sent_direct_3d", "sent_direct_5d",
        "sent_macro_1d", "sent_const_rollup_1d", "sent_composite_1d",
        "sent_neg_share_1d", "sent_dispersion_1d", "sent_count_1d",
        "sent_shock_count_1d", "sent_zscore_20d", "sent_change_1d",
        "sent_abs_mean_1d",
    ]
    return pd.DataFrame(0.0, index=index, columns=cols)
