"""
News loader: RSS ingestion and historical file loading for IBEX35 sentiment.

Sources (publicly available RSS, no auth required):
  Tier 1 — Spanish financial press: Expansión, Cinco Días, El Economista
  Tier 2 — General economic press: El País Economía, El Mundo Economía
  Tier 3 — Aggregators: Google News (ES + EN IBEX queries)

Output schema: DataFrame with columns
  [published, title, text, source, lang]
All timestamps are UTC-normalised.
Caching: parquet file per day in cache_dir.
"""
from __future__ import annotations

import hashlib
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

from src.utils.config import get, root
from src.utils.logging import get_logger

log = get_logger(__name__)

# ── RSS feed registry ─────────────────────────────────────────────────────────
# Each entry: (url, source_key, language)
_FEEDS: list[tuple[str, str, str]] = [
    # Spanish financial press
    ("https://www.expansion.com/rss/mercados.xml",                 "expansion",    "es"),
    ("https://cincodias.elpais.com/rss/cincodias/mercados/",        "cincodias",    "es"),
    ("https://www.eleconomista.es/rss/rss-bolsa-mercados.php",      "eleconomista", "es"),
    ("https://www.bolsamania.com/noticias/feed",                    "bolsamania",   "es"),
    ("https://www.invertia.com/es/rss/mercados",                    "invertia",     "es"),
    # General economic press
    ("https://feeds.elpais.com/mrss-s/pages/ep/site/elpais.com/section/economia/portada",
                                                                    "elpais",       "es"),
    # English / international
    ("https://feeds.reuters.com/reuters/businessNews",              "reuters",      "en"),
]

# Keywords to filter for IBEX-relevant articles
_FILTER_KEYWORDS_ES = frozenset([
    "ibex", "bolsa", "mercado", "accion", "acción", "bme", "madrid",
    "inditex", "bbva", "santander", "iberdrola", "ferrovial", "amadeus",
    "caixabank", "repsol", "telefonica", "cellnex", "aena", "naturgy",
    "acs", "endesa", "sabadell",
])
_FILTER_KEYWORDS_EN = frozenset([
    "ibex", "spain", "spanish", "madrid", "inditex", "bbva", "santander",
    "iberdrola", "ferrovial", "repsol", "telefonica",
])


def _is_relevant(text: str, lang: str) -> bool:
    tl = text.lower()
    kws = _FILTER_KEYWORDS_ES if lang == "es" else _FILTER_KEYWORDS_EN
    return any(kw in tl for kw in kws)


def _parse_feed_entry(entry, source: str, lang: str) -> Optional[dict]:
    """Parse a feedparser entry into a canonical dict."""
    pub = entry.get("published_parsed") or entry.get("updated_parsed")
    if pub is None:
        return None
    try:
        published = datetime(*pub[:6], tzinfo=timezone.utc)
    except (TypeError, ValueError):
        return None

    title   = entry.get("title", "").strip()
    summary = entry.get("summary", "").strip()
    text    = (title + " " + summary).strip()

    if not _is_relevant(text, lang):
        return None

    return {"published": published, "title": title, "text": summary,
            "source": source, "lang": lang}


def _cache_path(cache_dir: Path, date: datetime.date) -> Path:
    return cache_dir / f"news_{date.isoformat()}.parquet"


def load_rss(
    days_back: int = 7,
    cache_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Fetch and parse RSS feeds.  Returns DataFrame with columns:
      [published, title, text, source, lang]

    Results are cached per-day to avoid redundant HTTP requests.
    """
    try:
        import feedparser  # type: ignore
    except ImportError:
        log.warning("feedparser not installed — RSS loading unavailable. "
                    "Run: pip install feedparser")
        return _empty_df()

    if cache_dir is None:
        cache_dir = root() / get("data.cache_dir", "data/cache") / "news"
    cache_dir.mkdir(parents=True, exist_ok=True)

    cutoff = datetime.now(timezone.utc) - timedelta(days=days_back)
    rows: list[dict] = []

    for url, source, lang in _FEEDS:
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries:
                row = _parse_feed_entry(entry, source, lang)
                if row and row["published"] >= cutoff:
                    rows.append(row)
        except Exception as exc:
            log.debug(f"RSS fetch failed [{source}]: {exc}")

    if not rows:
        log.warning("No RSS articles fetched")
        return _empty_df()

    df = pd.DataFrame(rows).drop_duplicates(subset=["title"])
    df["published"] = pd.to_datetime(df["published"], utc=True)
    log.info(f"RSS: fetched {len(df)} articles from {df['source'].nunique()} sources")
    return df.sort_values("published").reset_index(drop=True)


def load_historical(
    data_dir: Optional[Path] = None,
    file_patterns: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Load historical articles from local CSV/parquet files.

    Expected columns: [published, title, text, source, lang]
    The 'published' column will be coerced to UTC datetime.
    """
    if data_dir is None:
        data_dir = root() / get("data.raw_dir", "data/raw") / "news"

    patterns = file_patterns or ["*.parquet", "*.csv"]
    files = []
    for pat in patterns:
        files.extend(data_dir.glob(pat))

    if not files:
        log.info(f"No historical news files found in {data_dir}")
        return _empty_df()

    parts: list[pd.DataFrame] = []
    for f in sorted(files):
        try:
            df = pd.read_parquet(f) if f.suffix == ".parquet" else pd.read_csv(f)
            parts.append(df)
        except Exception as exc:
            log.debug(f"Could not read {f}: {exc}")

    if not parts:
        return _empty_df()

    out = pd.concat(parts, ignore_index=True)
    out["published"] = pd.to_datetime(out["published"], utc=True, errors="coerce")
    out = out.dropna(subset=["published"])

    required = {"title", "text", "source", "lang"}
    for col in required - set(out.columns):
        out[col] = "" if col != "lang" else "es"

    return out.sort_values("published").reset_index(drop=True)


def load_all(
    days_back: int = 90,
    cache_dir: Optional[Path] = None,
    data_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Combine RSS (recent) + historical (archive) into one sorted DataFrame."""
    rss  = load_rss(days_back=days_back, cache_dir=cache_dir)
    hist = load_historical(data_dir=data_dir)

    if hist.empty:
        return rss
    if rss.empty:
        return hist

    combined = pd.concat([hist, rss], ignore_index=True)
    combined = combined.drop_duplicates(subset=["title"]).sort_values("published")
    return combined.reset_index(drop=True)


def _empty_df() -> pd.DataFrame:
    return pd.DataFrame(columns=["published", "title", "text", "source", "lang"])
