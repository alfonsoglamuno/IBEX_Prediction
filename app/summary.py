"""
Operative Summary: explains the IBEX35 prediction in plain English.

Generates a structured report combining:
  1. Prediction signal       — direction, probability, horizon, model
  2. SHAP-driven narrative   — top features and what they say
  3. Sentiment state         — three-layer composite with interpretation
  4. Media section           — recent headlines with per-article sentiment
  5. Risk factors            — what could flip the current call

Used by: dashboard (tab), API (/summary), CLI (predict.py --summary)
"""
from __future__ import annotations

import json
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.utils.config import get, root
from src.utils.logging import get_logger

log = get_logger(__name__)

# ── SHAP feature interpretation table ────────────────────────────────────────
# Maps feature name prefixes to human-readable descriptions and directionality.
_FEAT_DESCRIPTIONS: dict[str, tuple[str, str]] = {
    # (description, direction_hint)  — direction_hint: "high=bullish" | "high=bearish" | "neutral"
    "ret_lag1":          ("Yesterday's log-return",               "high=bullish"),
    "ret_lag2":          ("2-day lagged return",                  "high=bullish"),
    "ret_lag5":          ("5-day lagged return",                  "high=bullish"),
    "ret_mean_5d":       ("5-day return momentum",                "high=bullish"),
    "ret_mean_21d":      ("21-day return momentum",               "high=bullish"),
    "ret_skew_21d":      ("21-day return skewness",               "neutral"),
    "sma20_dist":        ("Price distance from 20-day average",   "high=bullish"),
    "sma50_dist":        ("Price distance from 50-day average",   "high=bullish"),
    "sma200_dist":       ("Price distance from 200-day average",  "high=bullish"),
    "ema8_dist":         ("Price distance from fast EMA (8d)",    "high=bullish"),
    "ema21_dist":        ("Price distance from mid EMA (21d)",    "high=bullish"),
    "cross_5_20":        ("5/20 SMA crossover strength",          "high=bullish"),
    "cross_20_50":       ("20/50 SMA crossover strength",         "high=bullish"),
    "cross_50_200":      ("50/200 SMA crossover (golden/death)",  "high=bullish"),
    "adx14":             ("Trend strength (ADX 14)",              "neutral"),
    "rsi14":             ("RSI 14 — momentum oscillator",         "neutral"),
    "macd_hist":         ("MACD histogram (momentum change)",     "high=bullish"),
    "bb_pct_b":          ("Bollinger %B — position within bands", "neutral"),
    "hv_21d":            ("21-day historical volatility",         "high=bearish"),
    "hv_63d":            ("63-day historical volatility",         "high=bearish"),
    "rv_21":             ("21-day realised volatility",           "high=bearish"),
    "atr14_norm":        ("ATR 14 (normalised) — daily range",    "high=bearish"),
    "vol_z20":           ("Volume z-score vs 20-day average",     "high=bullish"),
    "obv_slope5":        ("OBV slope — buying pressure trend",    "high=bullish"),
    "cmf20":             ("Chaikin Money Flow — funds flow",      "high=bullish"),
    "mfi14":             ("Money Flow Index",                     "high=bullish"),
    "vix_level":         ("VIX level — US fear index",            "high=bearish"),
    "vix_z21":           ("VIX z-score vs 21-day mean",          "high=bearish"),
    "stoxx50_lag1":      ("EURO STOXX 50 yesterday's return",    "high=bullish"),
    "sp500_lag1":        ("S&P 500 yesterday's return",          "high=bullish"),
    "ibex_vs_stoxx_rs5": ("IBEX relative strength vs STOXX (5d)","high=bullish"),
    "sent_composite_1d": ("News sentiment composite (yesterday)", "high=bullish"),
    "sent_direct_1d":    ("Direct IBEX news sentiment",          "high=bullish"),
    "sent_macro_1d":     ("Macro / ECB sentiment signal",        "high=bullish"),
    "sent_const_rollup_1d": ("Constituent news roll-up",         "high=bullish"),
    "sent_zscore_20d":   ("Sentiment z-score vs 20d mean",       "high=bullish"),
    "sent_neg_share_1d": ("Share of negative headlines",         "high=bearish"),
}


def _describe_feature(name: str, shap_value: float, feature_value: float) -> str:
    """Return a plain-English sentence describing a SHAP feature contribution."""
    # Find matching description
    desc, direction = "Unknown feature", "neutral"
    for prefix, (d, di) in _FEAT_DESCRIPTIONS.items():
        if name.startswith(prefix) or name == prefix:
            desc, direction = d, di
            break

    # Direction of push
    direction_of_shap = "pushes UP" if shap_value > 0 else "pushes DOWN"
    magnitude = abs(shap_value)
    strength = "strongly" if magnitude > 0.05 else "moderately" if magnitude > 0.02 else "slightly"

    # Contextual value description
    if not np.isnan(feature_value):
        val_str = f"(current value: {feature_value:.3f})"
    else:
        val_str = ""

    return f"**{desc}** {val_str} — {strength} {direction_of_shap}"


def _sentiment_label(score: float) -> str:
    if score > 0.20:   return "strongly positive"
    if score > 0.05:   return "slightly positive"
    if score < -0.20:  return "strongly negative"
    if score < -0.05:  return "slightly negative"
    return "neutral"


def _signal_color_emoji(signal: str) -> str:
    return {"UP": "🟢", "DOWN": "🔴", "NEUTRAL": "🟡"}.get(signal, "⬜")


def _confidence_bar(prob: float, width: int = 20) -> str:
    filled = round(prob * width)
    return "█" * filled + "░" * (width - filled) + f"  {prob:.1%}"


# ── Sentiment data loader ─────────────────────────────────────────────────────

def _load_live_sentiment(days_back: int = 3) -> tuple[dict, list[dict]]:
    """
    Returns (daily_sentiment_dict, list_of_scored_articles).
    Falls back to zeros if news loading fails.
    """
    try:
        from src.data.news_loader import load_rss
        from src.nlp.sentiment_scoring import score_batch
        from src.data.news_entities import classify_article

        df_news = load_rss(days_back=days_back)
        if df_news.empty:
            return {}, []

        articles_raw = df_news.to_dict("records")
        scored = score_batch(articles_raw, lang=get("sentiment.lang", "es"))

        # Build per-layer aggregates from today / yesterday
        relevant = [a for a in scored if a.target_type != "irrelevant"]

        def wmean(group):
            if not group:
                return 0.0
            scores  = [a.sentiment.score for a in group]
            weights = [a.combined_weight for a in group]
            total = sum(weights)
            return sum(s*w for s, w in zip(scores, weights)) / total if total > 0 else 0.0

        direct  = [a for a in relevant if a.target_type == "index"]
        const   = [a for a in relevant if a.target_type == "constituent"]
        macro   = [a for a in relevant if a.target_type == "macro"]

        alpha = get("sentiment.alpha", 0.50)
        beta  = get("sentiment.beta",  0.30)
        gamma = get("sentiment.gamma", 0.20)

        direct_s = wmean(direct)
        const_s  = wmean([a.__class__(
            title=a.title, text=a.text, published=a.published, source=a.source,
            target_type=a.target_type, target_name=a.target_name,
            relevance_score=a.relevance_score,
            constituent_weight=a.constituent_weight,
            sentiment=a.sentiment,
            source_credibility=a.source_credibility,
            time_weight=a.time_weight, novelty_weight=a.novelty_weight,
        ) for a in const] if const else [])
        const_s  = wmean(const) if const else 0.0
        macro_s  = wmean(macro)
        composite = alpha * direct_s + beta * const_s + gamma * macro_s

        sentiment_state = {
            "composite":       round(composite, 4),
            "direct_ibex":     round(direct_s, 4),
            "constituent_rollup": round(const_s, 4),
            "macro":           round(macro_s, 4),
            "label":           _sentiment_label(composite),
            "article_count":   len(relevant),
            "neg_share":       round(sum(1 for a in relevant if a.sentiment.score < -0.05)
                                     / max(1, len(relevant)), 3),
        }

        # Top articles for media section
        media = []
        for a in sorted(relevant, key=lambda x: abs(x.sentiment.score), reverse=True)[:15]:
            media.append({
                "published":       a.published.strftime("%Y-%m-%d %H:%M UTC") if a.published else "",
                "source":          a.source,
                "headline":        a.title,
                "target_type":     a.target_type,
                "target_name":     a.target_name or "—",
                "sentiment_score": round(a.sentiment.score, 3),
                "sentiment_label": a.sentiment.label,
                "confidence":      round(a.sentiment.confidence, 2),
                "relevance":       round(a.relevance_score, 2),
                "weight":          round(a.combined_weight, 3),
            })

        return sentiment_state, media

    except Exception as exc:
        log.debug(f"Live sentiment failed: {exc}")
        return {}, []


# ── Main summary generator ────────────────────────────────────────────────────

def generate_summary(
    prediction_path: Optional[Path] = None,
    include_live_news: bool = True,
    n_shap_features: int = 6,
) -> dict:
    """
    Generate the full operative summary.

    Returns a dict with keys:
      generated_at, prediction, narrative_md, shap_drivers,
      sentiment, media, risk_factors
    """
    pred_path = prediction_path or root() / "results" / "latest_prediction.json"

    # ── 1. Load prediction ────────────────────────────────────────────────────
    pred: dict = {}
    if pred_path.exists():
        with open(pred_path) as f:
            pred = json.load(f)

    has_pred = bool(pred)
    signal    = pred.get("signal", "UNKNOWN")
    prob      = pred.get("probability", None)
    horizon   = pred.get("horizon", "1d")
    model     = pred.get("model", "unknown")
    pred_date = pred.get("date", "—")
    shap_raw  = pred.get("shap_values", {})

    # ── 2. Top SHAP drivers ───────────────────────────────────────────────────
    feat_vals = pred.get("feature_values", {})
    shap_sorted = sorted(shap_raw.items(), key=lambda x: abs(x[1]), reverse=True)[:n_shap_features]
    shap_drivers = []
    for name, sv in shap_sorted:
        fv = feat_vals.get(name, float("nan"))
        shap_drivers.append({
            "feature":       name,
            "shap_value":    round(sv, 5),
            "feature_value": round(fv, 5) if not np.isnan(fv) else None,
            "direction":     "UP" if sv > 0 else "DOWN",
            "sentence":      _describe_feature(name, sv, fv),
        })

    # ── 3. Live sentiment ─────────────────────────────────────────────────────
    sentiment_state: dict = {}
    media_articles: list[dict] = []
    if include_live_news:
        try:
            sentiment_state, media_articles = _load_live_sentiment(days_back=2)
        except Exception as exc:
            log.debug(f"Sentiment load failed: {exc}")

    # ── 4. Risk factors ───────────────────────────────────────────────────────
    risk_factors = _build_risk_factors(signal, prob, shap_drivers, sentiment_state)

    # ── 5. Narrative markdown ─────────────────────────────────────────────────
    narrative_md = _build_narrative(
        signal, prob, horizon, model, pred_date,
        shap_drivers, sentiment_state, media_articles, risk_factors
    )

    return {
        "generated_at":  datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "prediction":    {
            "signal":      signal,
            "probability": prob,
            "horizon":     horizon,
            "model":       model,
            "date":        pred_date,
        },
        "narrative_md":  narrative_md,
        "shap_drivers":  shap_drivers,
        "sentiment":     sentiment_state,
        "media":         media_articles,
        "risk_factors":  risk_factors,
    }


def _build_risk_factors(
    signal: str, prob: Optional[float],
    shap_drivers: list[dict], sentiment: dict
) -> list[str]:
    """Identify conditions that could invalidate the current signal."""
    risks = []

    if prob is not None and abs(prob - 0.5) < 0.07:
        risks.append("Low model conviction (probability near 50%) — signal is unreliable.")

    # Check if top driver is negative-sentiment feature
    if shap_drivers:
        top = shap_drivers[0]
        if "volatility" in top["feature"] or "vix" in top["feature"]:
            risks.append("Volatility is the primary driver — regime shifts can rapidly change the signal.")

    # Sentiment divergence
    if sentiment:
        if sentiment.get("neg_share", 0) > 0.5:
            risks.append("More than 50% of recent headlines are negative — sentiment risk is elevated.")
        composite = sentiment.get("composite", 0)
        if signal == "UP" and composite < -0.10:
            risks.append("Model is calling UP but news sentiment is negative — potential divergence.")
        if signal == "DOWN" and composite > 0.10:
            risks.append("Model is calling DOWN but news sentiment is positive — potential divergence.")

    if not risks:
        risks.append("No specific risk alerts identified. Standard market risk applies.")

    return risks


def _build_narrative(
    signal: str, prob: Optional[float], horizon: str, model: str, pred_date: str,
    shap_drivers: list[dict], sentiment: dict, media: list[dict], risks: list[str]
) -> str:
    """Build the full markdown narrative."""
    emoji = _signal_color_emoji(signal)
    lines: list[str] = []

    # ── Header ────────────────────────────────────────────────────────────────
    lines.append(f"## {emoji} Operative Summary — IBEX35 {horizon.upper()} Outlook")
    lines.append(f"*Generated: {datetime.now(timezone.utc).strftime('%d %b %Y, %H:%M UTC')}*")
    lines.append("")

    # ── Signal ────────────────────────────────────────────────────────────────
    lines.append("---")
    lines.append("### Prediction")
    if prob is not None:
        prob_bar = _confidence_bar(prob)
        lines.append(f"**Signal: {signal}** &nbsp; Probability: `{prob_bar}`")
        lines.append(f"Model: `{model}` · Horizon: `{horizon}` · As of: `{pred_date}`")

        # Plain-English signal interpretation
        horizon_label = "next trading day" if horizon == "1d" else "next 5 trading days"
        if signal == "UP" and prob >= 0.55:
            lines.append(f"\nThe model expects the IBEX35 to **rise over the {horizon_label}** "
                         f"with {prob:.0%} confidence. This exceeds the 55% confidence threshold for a long signal.")
        elif signal == "DOWN" and prob <= 0.45:
            lines.append(f"\nThe model expects the IBEX35 to **fall over the {horizon_label}** "
                         f"({1-prob:.0%} probability of a down move).")
        else:
            lines.append(f"\nThe model is **neutral** for the {horizon_label}. "
                         "Probability is near 50%, below the confidence threshold.")
    else:
        lines.append("*No prediction available. Run `python predict.py --shap` to generate.*")
    lines.append("")

    # ── SHAP drivers ─────────────────────────────────────────────────────────
    if shap_drivers:
        lines.append("---")
        lines.append("### Why? — Model Drivers (SHAP)")
        lines.append("The following features had the largest influence on today's prediction:\n")
        for i, d in enumerate(shap_drivers, 1):
            lines.append(f"{i}. {d['sentence']}")
        lines.append("")

    # ── Sentiment ─────────────────────────────────────────────────────────────
    lines.append("---")
    lines.append("### News Sentiment")
    if sentiment:
        comp  = sentiment.get("composite", 0)
        label = sentiment.get("label", "neutral")
        count = sentiment.get("article_count", 0)
        neg_s = sentiment.get("neg_share", 0)

        lines.append(f"**Composite sentiment: {comp:+.3f}** — {label.title()}")
        lines.append("")
        lines.append("| Layer | Score | Interpretation |")
        lines.append("|-------|-------|----------------|")
        lines.append(f"| Direct IBEX | `{sentiment.get('direct_ibex', 0):+.3f}` | "
                     f"{_sentiment_label(sentiment.get('direct_ibex', 0)).title()} |")
        lines.append(f"| Constituent roll-up | `{sentiment.get('constituent_rollup', 0):+.3f}` | "
                     f"{_sentiment_label(sentiment.get('constituent_rollup', 0)).title()} |")
        lines.append(f"| Macro / ECB | `{sentiment.get('macro', 0):+.3f}` | "
                     f"{_sentiment_label(sentiment.get('macro', 0)).title()} |")
        lines.append(f"| **Composite** (α·direct + β·rollup + γ·macro) | **`{comp:+.3f}`** | "
                     f"**{label.title()}** |")
        lines.append("")
        lines.append(f"Relevant articles: **{count}** · Negative share: **{neg_s:.0%}**")
        lines.append("")

        # Sentiment narrative
        if abs(comp) < 0.05:
            lines.append("News tone is broadly neutral — no strong directional signal from media.")
        elif comp > 0:
            lines.append(f"Media coverage is **positive**, primarily driven by "
                         f"{'IBEX-level market news' if sentiment.get('direct_ibex',0) > 0.05 else 'positive constituent news'}. "
                         "This is broadly consistent with an UP call.")
        else:
            lines.append(f"Media coverage is **negative**, "
                         f"{'with macro headwinds (ECB / rates)' if sentiment.get('macro',0) < -0.05 else 'with negative headlines on major stocks'}. "
                         "This is broadly consistent with a DOWN call.")
    else:
        lines.append("*Sentiment data unavailable. Check RSS connectivity or run with `include_sentiment: true`.*")
    lines.append("")

    # ── Media section ─────────────────────────────────────────────────────────
    if media:
        lines.append("---")
        lines.append("### Media Coverage")
        lines.append(f"*Top {min(10, len(media))} recent IBEX-relevant headlines by sentiment impact:*\n")

        EMOJI_LAYER = {"index": "📊", "constituent": "🏢", "macro": "🌍", "irrelevant": "—"}
        EMOJI_SENT  = {"pos": "🟢", "neu": "🟡", "neg": "🔴"}

        lines.append("| # | Source | Headline | Target | Sentiment |")
        lines.append("|---|--------|----------|--------|-----------|")
        for i, art in enumerate(media[:10], 1):
            layer_e = EMOJI_LAYER.get(art["target_type"], "—")
            sent_e  = EMOJI_SENT.get(art["sentiment_label"], "⬜")
            headline = art["headline"][:70] + ("…" if len(art["headline"]) > 70 else "")
            score    = art["sentiment_score"]
            lines.append(f"| {i} | `{art['source']}` | {headline} | "
                         f"{layer_e} {art['target_name']} | {sent_e} `{score:+.3f}` |")
        lines.append("")
        lines.append(f"*Published: {media[0]['published'] if media else '—'} — "
                     f"{media[-1]['published'] if len(media) > 1 else ''}*")
        lines.append("")

    # ── Risk factors ──────────────────────────────────────────────────────────
    lines.append("---")
    lines.append("### Risk Factors")
    for risk in risks:
        lines.append(f"- {risk}")
    lines.append("")

    # ── Disclaimer ────────────────────────────────────────────────────────────
    lines.append("---")
    lines.append("*This summary is generated by a statistical model. It is not financial advice. "
                 "Past performance does not guarantee future results.*")

    return "\n".join(lines)


def save_summary(summary: dict, path: Optional[Path] = None) -> Path:
    """Save the summary dict to JSON."""
    out_path = path or root() / "results" / "latest_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
    return out_path
