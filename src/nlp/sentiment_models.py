"""
Sentiment scoring models for Spanish/English financial text.

Model hierarchy (best → fallback):
  1. TransformerModel  — BETO-finance (ES) / FinBERT (EN) / multilingual (XX)
  2. KeywordModel      — Loughran-McDonald adapted financial lexicon (always available)

Literature basis:
  - Loughran & McDonald (2011): domain-specific financial dictionaries outperform
    general-purpose lexicons (Harvard IV, LIWC) for finance.
  - Araci (2019): FinBERT trained on Financial PhraseBank; state-of-the-art English.
  - Barbieri et al. (2021): MarIA (PlanTL-GOB-ES/roberta-base-bne) — best Spanish BERT.
  - González-Carvajal & Garrido-Merchán (2021): BETO competitive on Spanish finance.
  - Malo et al. (2014): Financial PhraseBank — the standard fine-grained finance dataset.

All models return SentimentResult with continuous score ∈ [-1, 1],
label ∈ {neg, neu, pos}, and confidence ∈ [0, 1].
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Literal
import re

from src.utils.logging import get_logger

log = get_logger(__name__)

SentLabel = Literal["neg", "neu", "pos"]


@dataclass(frozen=True)
class SentimentResult:
    score: float        # continuous ∈ [-1, 1]
    label: SentLabel    # neg / neu / pos
    confidence: float   # ∈ [0, 1]

    @staticmethod
    def from_score(score: float, confidence: float = 1.0) -> "SentimentResult":
        label: SentLabel = "neu" if abs(score) < 0.05 else ("pos" if score > 0 else "neg")
        return SentimentResult(score=round(score, 4), label=label,
                               confidence=round(confidence, 4))

    @staticmethod
    def neutral() -> "SentimentResult":
        return SentimentResult(score=0.0, label="neu", confidence=0.0)


# ── Loughran-McDonald adapted financial lexicon ───────────────────────────────
# Spanish terms drawn from: Loughran & McDonald (2011) + FiQA lexicon translated
# + Financial PhraseBank (Malo et al. 2014) translated into ES.
# English terms are the original L&M wordlists.

_POS_ES: frozenset[str] = frozenset({
    # Price / market movement
    "sube", "subida", "suba", "rebota", "rebote", "avanza", "avance", "rally",
    "alcista", "alza", "máximo", "record", "récord", "supera", "superar",
    "dispararse", "repunta", "repunte",
    # Earnings / results
    "gana", "ganancias", "beneficio", "beneficios", "rentabilidad", "dividendo",
    "beneficio neto", "margen", "crecimiento", "mejora", "supera previsiones",
    "bate expectativas", "supera estimaciones", "record de beneficios",
    "mejores resultados", "resultados positivos",
    # Economy / macro
    "crecimiento", "expansión", "recuperación", "fortaleza", "impulso",
    "optimismo", "confianza", "compra", "compras", "demanda",
    # Corporate
    "acuerdo", "fusión", "adquisición", "contrato", "inversión",
    "aceleración", "expansión internacional", "aumento de capital",
})

_NEG_ES: frozenset[str] = frozenset({
    # Price / market movement
    "baja", "bajada", "baje", "cae", "caída", "desplome", "retrocede", "retroceso",
    "bajista", "mínimo", "pérdidas", "pérdida", "corrección", "desploma",
    "hundimiento", "hunde", "derrumbe", "derrumba",
    # Earnings / results
    "pierde", "pérdida neta", "déficit", "deterioro", "decepciona",
    "por debajo de previsiones", "incumple expectativas", "recorta dividendo",
    "resultados negativos", "caída de beneficios", "beneficios caen",
    # Economy / macro
    "recesión", "crisis", "contracción", "debilidad", "tensión",
    "pesimismo", "incertidumbre", "riesgo", "deuda",
    # Corporate
    "quiebra", "concurso de acreedores", "reestructuración", "despidos",
    "expediente de regulación", "ere", "multa", "sanción", "demanda judicial",
    "investigación regulatoria", "profit warning", "advertencia de beneficios",
})

_INTENSIFIERS_ES: frozenset[str] = frozenset({
    "muy", "fuerte", "gran", "grande", "enorme", "notable", "significativo",
    "importante", "histórico", "máximo histórico", "récord histórico",
    "excepcional", "extraordinario",
})

_NEGATORS_ES: frozenset[str] = frozenset({
    "no", "ni", "sin", "nunca", "jamás", "apenas",
})

_POS_EN: frozenset[str] = frozenset({
    "gain", "gains", "rally", "surge", "surges", "rise", "rises", "climbs",
    "jumps", "soars", "record", "high", "profit", "profits", "beat", "beats",
    "outperforms", "exceeds", "growth", "recovery", "strong", "bullish",
    "upgrade", "buy", "positive", "optimism", "deal", "acquisition", "merger",
    "dividend", "expansion",
})

_NEG_EN: frozenset[str] = frozenset({
    "fall", "falls", "drop", "drops", "decline", "declines", "slump", "slumps",
    "plunge", "plunges", "crash", "loss", "losses", "miss", "misses", "below",
    "weak", "bearish", "downgrade", "sell", "negative", "pessimism", "debt",
    "recession", "crisis", "bankruptcy", "layoffs", "restructuring", "fine",
    "probe", "investigation", "warning", "profit warning",
})

_NEGATORS_EN: frozenset[str] = frozenset({
    "not", "no", "never", "without", "hardly", "barely",
})


class KeywordModel:
    """
    Fast rule-based sentiment using Loughran-McDonald adapted financial lexicon.

    Scoring:
      - Tokenise text into words
      - Check for negation in a window of ±2 tokens around each sentiment word
      - Intensifiers multiply the polarity by 1.5
      - score = (pos_sum - neg_sum) / (pos_sum + neg_sum + ε)
    """

    def score(self, text: str, lang: str = "es") -> SentimentResult:
        pos_kws = _POS_ES if lang == "es" else _POS_EN
        neg_kws = _NEG_ES if lang == "es" else _NEG_EN
        neg_tokens = _NEGATORS_ES if lang == "es" else _NEGATORS_EN

        tokens = re.findall(r"\b\w+\b", text.lower())
        token_set = set(tokens)

        pos_score = neg_score = 0.0

        for i, tok in enumerate(tokens):
            window_start = max(0, i - 2)
            window = set(tokens[window_start:i])
            negated = bool(window & neg_tokens)
            intensified = bool(window & _INTENSIFIERS_ES)
            multiplier = 1.5 if intensified else 1.0

            if tok in pos_kws or any(tok.startswith(kw) for kw in pos_kws if len(kw) > 4):
                if negated:
                    neg_score += multiplier
                else:
                    pos_score += multiplier
            elif tok in neg_kws or any(tok.startswith(kw) for kw in neg_kws if len(kw) > 4):
                if negated:
                    pos_score += multiplier * 0.5  # negated negative → weak positive
                else:
                    neg_score += multiplier

        total = pos_score + neg_score
        if total == 0:
            return SentimentResult.neutral()

        raw = (pos_score - neg_score) / total
        confidence = min(1.0, total / 5.0)  # saturates at 5 sentiment words
        return SentimentResult.from_score(raw, confidence)

    def score_batch(self, texts: list[str], lang: str = "es") -> list[SentimentResult]:
        return [self.score(t, lang) for t in texts]


class TransformerModel:
    """
    HuggingFace transformer-based sentiment model.

    Model priority:
      Spanish:      PlanTL-GOB-ES/roberta-base-bne (MarIA) or
                    dccuchile/bert-base-spanish-wwm-cased (BETO) fine-tuned
      English:      ProsusAI/finbert
      Multilingual: nlptown/bert-base-multilingual-uncased-sentiment

    Lazy-loaded: the pipeline is initialised on first use.
    """

    _MODEL_IDS = {
        "es": [
            "mrm8488/bert-spanish-cased-finetuned-financial-news-sentiment-analysis",
            "dccuchile/bert-base-spanish-wwm-cased",
        ],
        "en": ["ProsusAI/finbert"],
        "xx": ["nlptown/bert-base-multilingual-uncased-sentiment"],
    }

    def __init__(self, lang: str = "es"):
        self._lang = lang
        self._pipe = None

    def _load(self) -> None:
        from transformers import pipeline as hf_pipeline  # type: ignore

        candidates = self._MODEL_IDS.get(self._lang, []) + self._MODEL_IDS["xx"]
        for model_id in candidates:
            try:
                self._pipe = hf_pipeline(
                    "text-classification",
                    model=model_id,
                    tokenizer=model_id,
                    top_k=None,
                    truncation=True,
                    max_length=512,
                )
                log.info(f"Loaded transformer model: {model_id}")
                return
            except Exception as exc:
                log.debug(f"Could not load {model_id}: {exc}")

        raise RuntimeError("No transformer model could be loaded. "
                           "Install transformers and a model checkpoint.")

    def _parse(self, raw: list[dict]) -> SentimentResult:
        # raw is list of {label: str, score: float}
        label_map: dict[str, float] = {}
        for item in raw:
            lbl = item["label"].lower()
            label_map[lbl] = item["score"]

        # Normalise across different label conventions
        pos = label_map.get("positive", label_map.get("pos",
              label_map.get("label_4", label_map.get("label_5", 0.0))))
        neg = label_map.get("negative", label_map.get("neg",
              label_map.get("label_1", label_map.get("label_2", 0.0))))
        neu = label_map.get("neutral", label_map.get("neu",
              label_map.get("label_3", 0.0)))

        # If model uses star labels (1-5), recalibrate
        if not (pos + neg + neu):
            total = sum(label_map.values())
            if total > 0:
                pos = label_map.get("label_5", 0) / total
                neg = label_map.get("label_1", 0) / total
                neu = 1 - pos - neg

        score = pos - neg
        confidence = max(pos, neg, neu)
        return SentimentResult.from_score(score, confidence)

    def score(self, text: str, lang: str | None = None) -> SentimentResult:
        if self._pipe is None:
            self._load()
        try:
            raw = self._pipe(text[:512])
            # pipeline with top_k=None returns list of dicts
            if isinstance(raw[0], list):
                raw = raw[0]
            return self._parse(raw)
        except Exception as exc:
            log.debug(f"Transformer scoring failed: {exc}")
            return SentimentResult.neutral()

    def score_batch(self, texts: list[str], lang: str | None = None,
                    batch_size: int = 32) -> list[SentimentResult]:
        if self._pipe is None:
            self._load()
        results = []
        for i in range(0, len(texts), batch_size):
            batch = [t[:512] for t in texts[i:i + batch_size]]
            try:
                raw_batch = self._pipe(batch)
                for raw in raw_batch:
                    if isinstance(raw, list):
                        results.append(self._parse(raw))
                    else:
                        results.append(self._parse([raw]))
            except Exception as exc:
                log.debug(f"Batch {i} failed: {exc}")
                results.extend([SentimentResult.neutral()] * len(batch))
        return results


@lru_cache(maxsize=4)
def get_model(lang: str = "es", backend: str = "auto") -> KeywordModel | TransformerModel:
    """
    Return the best available sentiment model.

    backend:
      "auto"        — try transformer first, fall back to keyword
      "transformer" — transformer only (raises if unavailable)
      "keyword"     — keyword model only (always available)
    """
    if backend == "keyword":
        return KeywordModel()

    if backend == "transformer":
        m = TransformerModel(lang=lang)
        m._load()
        return m

    # auto: try transformer, fall back silently
    try:
        import transformers  # noqa: F401
        m = TransformerModel(lang=lang)
        m._load()
        return m
    except Exception:
        log.info("transformers unavailable or model load failed — using keyword model")
        return KeywordModel()
