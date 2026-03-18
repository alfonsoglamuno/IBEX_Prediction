"""
Daily sentiment feature construction for IBEX35 forecasting.

Three-layer architecture (Tetlock 2007; Loughran & McDonald 2011; Baker & Wurgler 2007):
  Layer 1 — Direct IBEX    (articles mentioning IBEX35 / bolsa española)
  Layer 2 — Constituent    (company articles, weighted by index weight)
  Layer 3 — Macro EU/Spain (ECB, rates, eurozone)

Composite: IndexSent_t = α·DirectIBEXSent_t + β·ConstituentRollup_t + γ·MacroSpainEU_t
Per-article weight: w_i = w_rel · w_src · w_time · w_novelty

───────────────────────────────────────────────────────────────
TWO-TRACK DESIGN (coverage integrity principle)

  The critical distinction:
    - "No relevant news today" → signal = 0  (neutral, a real observation)
    - "Archive not available"  → signal = NaN (missing, not a real observation)

  Implementation:
    - sentiment_start_date (config): first date with trustworthy RSS/archive coverage
    - Dates BEFORE sentiment_start_date → NaN  (archive unavailable → missing)
    - Dates AFTER start_date, no news  → 0     (covered day, no relevant articles)

  Consequence for walk-forward:
    - include_sentiment: false  → Track A: full market history, no sentiment cols
    - include_sentiment: true   → Track B: sentiment-era only (rows before
      sentiment_start_date are dropped by build_features dropna)

  This is the only fair way to test whether sentiment adds incremental value:
  same folds, same dates, base vs base+sentiment.
  Reference: IBEX-35 news-emotions paper; Tetlock (2007); missing-variable literature.

───────────────────────────────────────────────────────────────
Feature block (all lagged 1 trading day — strictly no lookahead):
  sent_composite_1d   — hybrid composite (α·direct + β·rollup + γ·macro)
  sent_direct_1d      — direct-IBEX weighted sentiment
  sent_direct_3d      — 3d EMA of direct-IBEX sentiment
  sent_direct_5d      — 5d EMA of direct-IBEX sentiment
  sent_macro_1d       — macro layer
  sent_const_rollup_1d— constituent roll-up
  sent_neg_share_1d   — share of negative articles
  sent_dispersion_1d  — std of scores (disagreement / uncertainty)
  sent_count_1d       — log(1 + article count)
  sent_shock_count_1d — count |score| > 0.5 (high-signal events)
  sent_zscore_20d     — composite z-score vs 20d mean
  sent_change_1d      — 1d change in composite
  sent_abs_mean_1d    — mean |score| (absolute intensity)
  sent_is_available   — 1 if coverage exists, 0 if not (use as control)

Literature:
  Tetlock (2007): media pessimism predicts next-day Dow Jones returns.
  Tetlock, Saar-Tsechansky & Macskassy (2008): negative words predict earnings.
  Loughran & McDonald (2011): finance-specific lexicon outperforms general-purpose.
  Baker & Wurgler (2007): investor sentiment and cross-section of stock returns.
  Groß-Klußmann & Hautsch (2011): news intensity and limit order book dynamics.
  Da, Engelberg & Gao (2011): news decay half-life ~1 day in sample.
  Garcia (2013): newspaper tone predicts returns, especially in recessions.
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

_ALPHA = 0.50   # direct IBEX
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
    """Compute all sentiment signals for a single covered trading day."""
    direct   = [a for a in articles if a.target_type == "index"]
    const    = [a for a in articles if a.target_type == "constituent"]
    macro    = [a for a in articles if a.target_type == "macro"]
    relevant = direct + const + macro

    def wmean(group: list[ScoredArticle]) -> float:
        return _safe_weighted_mean(
            [a.sentiment.score for a in group],
            [a.combined_weight for a in group],
        )

    direct_sent = wmean(direct)
    macro_sent  = wmean(macro)
    const_rollup = _safe_weighted_mean(
        [a.sentiment.score for a in const],
        [a.combined_weight * a.constituent_weight for a in const],
    ) if const else 0.0

    alpha = get("sentiment.alpha", _ALPHA)
    beta  = get("sentiment.beta",  _BETA)
    gamma = get("sentiment.gamma", _GAMMA)
    composite = alpha * direct_sent + beta * const_rollup + gamma * macro_sent

    all_scores  = [a.sentiment.score for a in relevant]
    neg_share   = sum(1 for s in all_scores if s < -0.05) / max(1, len(all_scores))
    dispersion  = float(np.std(all_scores)) if len(all_scores) > 1 else 0.0
    shock_count = sum(1 for s in all_scores if abs(s) > 0.5)
    count_log   = float(np.log1p(len(relevant)))

    return {
        "sent_composite_1d":    composite,
        "sent_direct_1d":       direct_sent,
        "sent_macro_1d":        macro_sent,
        "sent_const_rollup_1d": const_rollup,
        "sent_neg_share_1d":    neg_share,
        "sent_dispersion_1d":   dispersion,
        "sent_count_1d":        count_log,
        "sent_shock_count_1d":  float(shock_count),
        "sent_abs_mean_1d":     float(np.mean(np.abs(all_scores))) if all_scores else 0.0,
    }


def build_sentiment_features(
    ibex_index: pd.DatetimeIndex,
    days_back: int = 365,
    lang: str = "es",
    backend: str = "auto",
) -> pd.DataFrame:
    """
    Build the full sentiment feature block aligned to ibex_index.

    Coverage gating:
      - Dates before `sentiment.start_date` (config): NaN  → "archive unavailable"
      - Dates within coverage, no articles:           0    → "no relevant news"
      - All values are then lagged by 1 trading day (no lookahead).

    The NaN rows before start_date will be dropped by build_features' dropna,
    implementing the two-track design described in this module's docstring.
    """
    start_date_str = get("sentiment.start_date", None)
    start_ts = pd.Timestamp(start_date_str) if start_date_str else None

    days_cfg   = get("sentiment.days_back", days_back)
    lang_cfg   = get("sentiment.lang", lang)
    backend_cfg = get("sentiment.backend", backend)

    log.info("Loading news articles...")
    df_news = load_all(days_back=days_cfg)

    # Build per-day aggregates from actual articles
    day_data: dict[str, dict] = {}

    if not df_news.empty:
        articles_raw = df_news.to_dict("records")
        scored: list[ScoredArticle] = score_batch(
            articles_raw, lang=lang_cfg, backend=backend_cfg
        )
        day_map: dict[str, list[ScoredArticle]] = {}
        for art in scored:
            d = art.published.astimezone(timezone.utc).date().isoformat()
            day_map.setdefault(d, []).append(art)

        for date_str, arts in day_map.items():
            day_data[date_str] = _aggregate_day(arts)

    # ── Align to ibex_index ───────────────────────────────────────────────────
    sent_cols = [
        "sent_composite_1d", "sent_direct_1d", "sent_macro_1d",
        "sent_const_rollup_1d", "sent_neg_share_1d", "sent_dispersion_1d",
        "sent_count_1d", "sent_shock_count_1d", "sent_abs_mean_1d",
    ]

    out = pd.DataFrame(np.nan, index=ibex_index, columns=sent_cols + ["sent_is_available"])

    for ts in ibex_index:
        date_str = ts.date().isoformat()

        # Before coverage start → NaN (archive unavailable)
        if start_ts is not None and ts < start_ts:
            continue   # leave as NaN

        # Within coverage
        out.loc[ts, "sent_is_available"] = 1.0
        if date_str in day_data:
            for col, val in day_data[date_str].items():
                out.loc[ts, col] = val
        else:
            # Covered day but no relevant articles → neutral (0)
            for col in sent_cols:
                out.loc[ts, col] = 0.0

    # Derived rolling features (computed on covered period only)
    composite = out["sent_composite_1d"]
    out["sent_direct_3d"] = composite.ewm(span=3, min_periods=1).mean()
    out["sent_direct_5d"] = composite.ewm(span=5, min_periods=1).mean()

    roll20 = composite.rolling(20, min_periods=5)
    out["sent_zscore_20d"] = (composite - roll20.mean()) / (roll20.std().replace(0, np.nan))
    out["sent_change_1d"]  = composite.diff(1)

    # Lag all features by 1 trading day — no lookahead
    lag_cols = [c for c in out.columns if c != "sent_is_available"]
    out[lag_cols] = out[lag_cols].shift(1)
    # sent_is_available is also lagged: we only know yesterday's sentiment
    out["sent_is_available"] = out["sent_is_available"].shift(1)

    # Forward-fill within the covered period only (news effect persists)
    if start_ts is not None:
        covered = out.index >= start_ts
        out.loc[covered] = out.loc[covered].ffill()

    n_covered = int((out["sent_is_available"] == 1).sum())
    n_nonzero = int((out["sent_composite_1d"].fillna(0) != 0).sum())
    log.info(f"Sentiment: {out.shape[1]} features | "
             f"{n_covered} covered days | {n_nonzero} non-zero signal days")
    return out


def _zero_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    cols = [
        "sent_composite_1d", "sent_direct_1d", "sent_direct_3d", "sent_direct_5d",
        "sent_macro_1d", "sent_const_rollup_1d", "sent_neg_share_1d",
        "sent_dispersion_1d", "sent_count_1d", "sent_shock_count_1d",
        "sent_zscore_20d", "sent_change_1d", "sent_abs_mean_1d", "sent_is_available",
    ]
    return pd.DataFrame(np.nan, index=index, columns=cols)
