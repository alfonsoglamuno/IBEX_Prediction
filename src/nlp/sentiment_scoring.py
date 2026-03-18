"""
Article-level sentiment scoring with entity classification.

Combines:
  - news_entities.classify_article()  → target_type, relevance, constituent weight
  - sentiment_models.get_model()      → continuous sentiment score
  - aspect-based weighting            → per-article weight w_i
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from src.data.news_entities import classify_article
from src.nlp.sentiment_models import SentimentResult, get_model
from src.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class ScoredArticle:
    """Article after entity classification and sentiment scoring."""
    title: str
    text: str
    published: datetime
    source: str

    # Filled by score_article()
    target_type: str = "irrelevant"
    target_name: Optional[str] = None
    relevance_score: float = 0.0
    constituent_weight: float = 0.0
    sentiment: SentimentResult = field(default_factory=SentimentResult.neutral)
    source_credibility: float = 0.6
    time_weight: float = 1.0
    novelty_weight: float = 1.0

    @property
    def combined_weight(self) -> float:
        """w_i = w_rel · w_src · w_time · w_novelty (Eq. 1, see sentiment_features.py)."""
        return (self.relevance_score
                * self.source_credibility
                * self.time_weight
                * self.novelty_weight)

    @property
    def weighted_score(self) -> float:
        return self.combined_weight * self.sentiment.score


# ── Source credibility table ──────────────────────────────────────────────────
# Literature: Groß-Klußmann & Hautsch (2011) show institutional wire services
# have faster and more accurate price discovery than retail aggregators.
SOURCE_CREDIBILITY: dict[str, float] = {
    # Wire / institutional
    "reuters":      1.00,
    "bloomberg":    1.00,
    "ft":           0.95,
    "wsj":          0.95,
    # Spanish financial press
    "expansion":    0.85,
    "cincodias":    0.85,
    "eleconomista": 0.80,
    "invertia":     0.78,
    "bolsamania":   0.75,
    # General press
    "elpais":       0.65,
    "elmundo":      0.65,
    "abc":          0.60,
    # Aggregators / default
    "google":       0.55,
    "default":      0.55,
}


def get_source_credibility(source: str) -> float:
    """Lookup source credibility; fuzzy match on source string."""
    s = source.lower()
    for key, w in SOURCE_CREDIBILITY.items():
        if key in s:
            return w
    return SOURCE_CREDIBILITY["default"]


def compute_time_weight(published: datetime, reference: datetime,
                        half_life_hours: float = 18.0) -> float:
    """
    Exponential decay: w_time = exp(-λ·Δt),  λ = ln(2) / half_life.

    Literature: Tetlock (2007) and Da et al. (2011) find news effects are
    largely absorbed within 1 trading day; half-life of 18h gives ~50% weight
    at market close on the publication day and ~25% the next morning.
    """
    if published.tzinfo is None:
        published = published.replace(tzinfo=timezone.utc)
    if reference.tzinfo is None:
        reference = reference.replace(tzinfo=timezone.utc)
    delta_hours = max(0.0, (reference - published).total_seconds() / 3600)
    lam = math.log(2) / half_life_hours
    return math.exp(-lam * delta_hours)


def compute_novelty_weight(article_idx: int, total_on_topic: int) -> float:
    """
    Novelty discount: w_novelty = 1 / sqrt(rank),  rank = order of articles on topic.

    Literature: Chan (2003) shows that first coverage of a corporate event has
    much stronger price impact than follow-up articles (news momentum).
    """
    rank = max(1, article_idx + 1)
    return 1.0 / math.sqrt(rank)


def score_article(
    title: str,
    text: str,
    published: datetime,
    source: str,
    reference_time: Optional[datetime] = None,
    lang: str = "es",
    backend: str = "auto",
) -> ScoredArticle:
    """Score a single article and return a ScoredArticle."""
    combined = (title + " " + text).strip()
    entity = classify_article(combined)

    model = get_model(lang=lang, backend=backend)
    result: SentimentResult = model.score(combined, lang=lang)  # type: ignore[arg-type]

    ref = reference_time or datetime.now(timezone.utc)
    t_weight = compute_time_weight(published, ref)
    src_credibility = get_source_credibility(source)

    return ScoredArticle(
        title=title,
        text=text,
        published=published,
        source=source,
        target_type=entity["target_type"],
        target_name=entity["target_name"],
        relevance_score=entity["relevance_score"],
        constituent_weight=entity["constituent_weight"],
        sentiment=result,
        source_credibility=src_credibility,
        time_weight=t_weight,
        novelty_weight=1.0,  # set externally by score_batch after topic grouping
    )


def score_batch(
    articles: list[dict],
    reference_time: Optional[datetime] = None,
    lang: str = "es",
    backend: str = "auto",
) -> list[ScoredArticle]:
    """
    Score a list of article dicts and apply novelty weights.

    Each dict must have: title, text, published (datetime), source (str).
    Novelty weights are applied per (target_name, date) group.
    """
    ref = reference_time or datetime.now(timezone.utc)
    scored = [
        score_article(
            title=a.get("title", ""),
            text=a.get("text", ""),
            published=a["published"],
            source=a.get("source", "default"),
            reference_time=ref,
            lang=lang,
            backend=backend,
        )
        for a in articles
    ]

    # Apply novelty weights per (target_name, publication_date) group
    from collections import defaultdict
    topic_counter: dict[tuple, int] = defaultdict(int)
    for art in sorted(scored, key=lambda x: x.published):
        key = (art.target_name, art.published.date())
        rank = topic_counter[key]
        art.novelty_weight = compute_novelty_weight(rank, total_on_topic=1)
        topic_counter[key] += 1

    return scored
