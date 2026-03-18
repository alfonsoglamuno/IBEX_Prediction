"""
Sentiment features entry point — delegates to sentiment_features.py.

This module exists for backwards-compatibility with any code that imports
`from src.data.sentiment import fetch_sentiment_features`.
"""
from src.data.sentiment_features import build_sentiment_features


def fetch_sentiment_features(ibex_index, days_back: int = 365, **kwargs):
    return build_sentiment_features(ibex_index, days_back=days_back, **kwargs)
