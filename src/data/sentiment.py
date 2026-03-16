"""
V2 — News sentiment features for IBEX35 forecasting.

Phase 1 (active): RSS-based keyword sentiment from Spanish financial news.
Phase 2 (stub):   FinBERT / RoBERTa sentence-level sentiment (activate when needed).

Usage:
    from src.data.sentiment import fetch_sentiment_features
    sentiment_df = fetch_sentiment_features(ibex_index)
"""
from __future__ import annotations

import re
from datetime import datetime, timedelta, date
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.config import get, root
from src.utils.logging import get_logger

log = get_logger(__name__)

# ── Keyword dictionaries (Spanish financial news) ─────────────────────────────

_POSITIVE = {
    "sube", "subida", "gana", "ganancia", "rebote", "récord", "máximo",
    "alcista", "positivo", "recupera", "compra", "optimismo", "crecimiento",
    "beneficios", "expansión", "alza", "favorable", "mejora", "supera",
    "fuerte", "impulso", "rally", "avance", "avanza",
}
_NEGATIVE = {
    "baja", "bajada", "pierde", "pérdida", "caída", "cae", "mínimo",
    "bajista", "negativo", "crisis", "vende", "pesimismo", "recesión",
    "pérdidas", "contracción", "corrección", "deuda", "riesgo", "tensión",
    "débil", "desplome", "quiebra", "retrocede", "retroceso",
}

# ── RSS feeds (publicly available, no auth required) ──────────────────────────

_RSS_FEEDS = [
    "https://www.expansion.com/rss/mercados.xml",
    "https://cincodias.elpais.com/rss/cincodias/mercados/",
    "https://feeds.elpais.com/mrss-s/pages/ep/site/elpais.com/section/economia/portada",
]

_IBEX_KEYWORDS = ["ibex", "bolsa", "mercado", "acción", "acciones", "bme", "madrid"]


def _score_text(text: str) -> float:
    """Returns sentiment score in [-1, +1] based on keyword matching."""
    words = set(re.findall(r"\b\w+\b", text.lower()))
    pos = len(words & _POSITIVE)
    neg = len(words & _NEGATIVE)
    total = pos + neg
    if total == 0:
        return 0.0
    return (pos - neg) / total


def _fetch_rss_articles(days_back: int = 7) -> list[dict]:
    """Fetch recent articles from RSS feeds. Returns list of {date, score} dicts."""
    try:
        import feedparser
    except ImportError:
        log.warning("feedparser not installed — skipping RSS sentiment")
        return []

    cutoff = datetime.now() - timedelta(days=days_back)
    articles = []

    for url in _RSS_FEEDS:
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries:
                # Parse date
                pub = entry.get("published_parsed") or entry.get("updated_parsed")
                if pub is None:
                    continue
                pub_dt = datetime(*pub[:6])
                if pub_dt < cutoff:
                    continue

                text = (entry.get("title", "") + " " + entry.get("summary", "")).lower()

                # Filter for IBEX-related articles
                if not any(kw in text for kw in _IBEX_KEYWORDS):
                    continue

                score = _score_text(text)
                articles.append({"date": pub_dt.date(), "score": score})
        except Exception as e:
            log.debug(f"RSS fetch failed for {url}: {e}")

    return articles


def fetch_sentiment_features(
    ibex_index: pd.DatetimeIndex,
    days_back: int = 30,
) -> pd.DataFrame:
    """
    Returns a DataFrame aligned to ibex_index with sentiment features.
    Features are strictly lagged (no lookahead).
    """
    if not get("features.include_sentiment", False):
        log.info("Sentiment disabled in config (features.include_sentiment=false)")
        return pd.DataFrame(index=ibex_index)

    articles = _fetch_rss_articles(days_back=days_back)

    if not articles:
        log.warning("No sentiment articles fetched")
        return pd.DataFrame(index=ibex_index)

    df_art = pd.DataFrame(articles)
    df_art["date"] = pd.to_datetime(df_art["date"])
    daily_sent = df_art.groupby("date")["score"].agg(["mean", "count"])
    daily_sent.index = pd.to_datetime(daily_sent.index)

    out = pd.DataFrame(index=ibex_index)
    out["sent_score"]  = daily_sent["mean"].reindex(ibex_index, method="ffill")
    out["sent_count"]  = daily_sent["count"].reindex(ibex_index, fill_value=0)
    out["sent_ma5"]    = out["sent_score"].rolling(5).mean()

    # Shift by 1: we can only use yesterday's news to predict today
    out["sent_score"]  = out["sent_score"].shift(1)
    out["sent_count"]  = out["sent_count"].shift(1)
    out["sent_ma5"]    = out["sent_ma5"].shift(1)

    log.info(f"Sentiment: {out.shape[1]} features, {out['sent_score'].notna().sum()} non-null rows")
    return out


# ── V2 Phase 2 stub — FinBERT / RoBERTa ─────────────────────────────────────
# To activate:
#   1. pip install transformers torch
#   2. Set features.include_sentiment: true in config.yaml
#   3. Uncomment and complete the function below
#
# def fetch_finbert_sentiment(texts: list[str]) -> list[float]:
#     from transformers import pipeline
#     nlp = pipeline("sentiment-analysis",
#                    model="ProsusAI/finbert",
#                    tokenizer="ProsusAI/finbert")
#     results = nlp(texts, truncation=True, max_length=512)
#     scores = []
#     for r in results:
#         if r["label"] == "positive":
#             scores.append(r["score"])
#         elif r["label"] == "negative":
#             scores.append(-r["score"])
#         else:
#             scores.append(0.0)
#     return scores
