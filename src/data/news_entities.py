"""
IBEX35 entity knowledge base for news relevance and target classification.

Classifies a news article as:
  "index"       — directly about the IBEX35 or Spanish equity market
  "constituent" — about a specific IBEX35 company
  "macro"       — about ECB / rates / eurozone factors that affect Spanish equities
  "irrelevant"  — no clear Spanish equity relevance

Constituent weights: approximate free-float market-cap weights (Q1 2025 composition).
Source: BME / STOXX official factsheets.

Used downstream by sentiment_scoring.py to apply layer-specific aggregation:
  - index      articles → DirectIBEXSent_t   (α layer)
  - constituent articles → ConstituentRollup_t (β layer, weighted by index weight)
  - macro      articles → MacroSpainEU_t     (γ layer)
"""
from __future__ import annotations

# ── Direct IBEX35 / Spanish equity market ────────────────────────────────────
INDEX_KEYWORDS: list[str] = [
    "ibex", "ibex 35", "ibex35",
    "bolsa española", "bolsa de madrid", "bolsa madrid",
    "mercado español", "renta variable española",
    "bme", "bolsas y mercados",
    "indice ibex", "benchmark español",
    "ibex continuo", "mercado continuo",
    "spanish stock market", "madrid stock exchange",
]

# ── Macro / systemic factors relevant to Spain ────────────────────────────────
# Literature: Garcia (2013) shows macro sentiment has incremental predictive power
# beyond equity-specific sentiment, especially during high-uncertainty regimes.
MACRO_KEYWORDS: list[str] = [
    # ECB / monetary policy
    "bce", "banco central europeo", "ecb", "european central bank",
    "tipos de interés", "subida de tipos", "bajada de tipos", "tipo de interés",
    "política monetaria", "reunión del bce", "decisión de tipos",
    # Inflation / rates
    "inflación", "inflacion", "euribor", "ipc", "deflación",
    "prima de riesgo", "spread soberano", "deuda española", "bono español",
    "deuda soberana", "tesoro español",
    # Spanish macro
    "pib españa", "economia española", "recesión españa",
    "desempleo españa", "paro españa", "actividad económica",
    # Eurozone / EU
    "eurozona", "euro area", "zona euro", "union europea",
    "mecanismo europeo", "esm", "mede", "fondo de recuperación",
    # Global macro influencing Spain
    "fed", "reserva federal", "federal reserve",
    "guerra comercial", "aranceles", "trade war", "tariffs",
    "geopolítica", "geopolitical",
    # Volatility / risk indices
    "vstoxx", "vix", "euro stoxx", "stoxx 50",
    "volatilidad europea", "risk off", "risk on",
]

# ── IBEX35 constituents ───────────────────────────────────────────────────────
# Weights: free-float adjusted, approximate Q1 2025 (BME factsheet).
# Sectors: consistent with GICS classification.
IBEX35_CONSTITUENTS: dict[str, dict] = {
    "Inditex": {
        "tickers":  ["ITX", "ITX.MC"],
        "names_es": ["inditex", "zara", "bershka", "pull and bear", "massimo dutti",
                     "stradivarius", "oysho", "zara home"],
        "names_en": ["inditex", "zara", "amancio ortega"],
        "weight":   0.152,
        "sector":   "Consumer Discretionary",
    },
    "BBVA": {
        "tickers":  ["BBVA", "BBVA.MC"],
        "names_es": ["bbva", "banco bilbao vizcaya argentaria", "bbva bancomer",
                     "bbva mexico", "bbva turquia"],
        "names_en": ["bbva", "banco bilbao"],
        "weight":   0.118,
        "sector":   "Financials",
    },
    "Santander": {
        "tickers":  ["SAN", "SAN.MC"],
        "names_es": ["santander", "banco santander", "grupo santander",
                     "santander brasil", "santander uk"],
        "names_en": ["santander", "banco santander"],
        "weight":   0.112,
        "sector":   "Financials",
    },
    "Iberdrola": {
        "tickers":  ["IBE", "IBE.MC"],
        "names_es": ["iberdrola", "iberdrola renovables", "avangrid", "scottishpower"],
        "names_en": ["iberdrola", "avangrid", "scottishpower"],
        "weight":   0.098,
        "sector":   "Utilities",
    },
    "Ferrovial": {
        "tickers":  ["FER", "FER.MC", "FER.AS"],
        "names_es": ["ferrovial", "grupo ferrovial", "cintra"],
        "names_en": ["ferrovial", "cintra"],
        "weight":   0.057,
        "sector":   "Industrials",
    },
    "Amadeus": {
        "tickers":  ["AMS", "AMS.MC"],
        "names_es": ["amadeus", "amadeus it"],
        "names_en": ["amadeus", "amadeus it group"],
        "weight":   0.052,
        "sector":   "Information Technology",
    },
    "CaixaBank": {
        "tickers":  ["CABK", "CABK.MC"],
        "names_es": ["caixabank", "la caixa", "caixa bank", "bankia", "imagin"],
        "names_en": ["caixabank"],
        "weight":   0.043,
        "sector":   "Financials",
    },
    "Repsol": {
        "tickers":  ["REP", "REP.MC"],
        "names_es": ["repsol", "repsol ypf", "repsol sinopec"],
        "names_en": ["repsol"],
        "weight":   0.041,
        "sector":   "Energy",
    },
    "Telefonica": {
        "tickers":  ["TEF", "TEF.MC"],
        "names_es": ["telefonica", "telefónica", "movistar", "o2", "vivo"],
        "names_en": ["telefonica", "movistar", "o2 spain"],
        "weight":   0.039,
        "sector":   "Communication Services",
    },
    "Cellnex": {
        "tickers":  ["CLNX", "CLNX.MC"],
        "names_es": ["cellnex", "cellnex telecom"],
        "names_en": ["cellnex"],
        "weight":   0.031,
        "sector":   "Communication Services",
    },
    "Aena": {
        "tickers":  ["AENA", "AENA.MC"],
        "names_es": ["aena", "aeropuertos españoles"],
        "names_en": ["aena", "spanish airports"],
        "weight":   0.030,
        "sector":   "Industrials",
    },
    "Naturgy": {
        "tickers":  ["NTGY", "NTGY.MC"],
        "names_es": ["naturgy", "gas natural", "gas natural fenosa"],
        "names_en": ["naturgy", "gas natural"],
        "weight":   0.028,
        "sector":   "Utilities",
    },
    "ACS": {
        "tickers":  ["ACS", "ACS.MC"],
        "names_es": ["acs", "grupo acs", "dragados", "hochtief"],
        "names_en": ["acs", "dragados", "hochtief"],
        "weight":   0.026,
        "sector":   "Industrials",
    },
    "Endesa": {
        "tickers":  ["ELE", "ELE.MC"],
        "names_es": ["endesa", "endesa españa"],
        "names_en": ["endesa"],
        "weight":   0.022,
        "sector":   "Utilities",
    },
    "Sabadell": {
        "tickers":  ["SAB", "SAB.MC"],
        "names_es": ["sabadell", "banco sabadell", "tsb bank"],
        "names_en": ["sabadell", "tsb"],
        "weight":   0.019,
        "sector":   "Financials",
    },
    "Mapfre": {
        "tickers":  ["MAP", "MAP.MC"],
        "names_es": ["mapfre"],
        "names_en": ["mapfre"],
        "weight":   0.013,
        "sector":   "Financials",
    },
    "IAG": {
        "tickers":  ["IAG", "IAG.MC", "IAG.L"],
        "names_es": ["iag", "international airlines group", "iberia", "vueling",
                     "british airways", "aer lingus"],
        "names_en": ["iag", "iberia", "british airways", "vueling"],
        "weight":   0.022,
        "sector":   "Industrials",
    },
    "Solaria": {
        "tickers":  ["SLR", "SLR.MC"],
        "names_es": ["solaria"],
        "names_en": ["solaria"],
        "weight":   0.008,
        "sector":   "Utilities",
    },
    "Acciona": {
        "tickers":  ["ANA", "ANA.MC"],
        "names_es": ["acciona", "acciona energia", "nordex"],
        "names_en": ["acciona"],
        "weight":   0.017,
        "sector":   "Industrials",
    },
    "Colonial": {
        "tickers":  ["COL", "COL.MC"],
        "names_es": ["colonial", "inmobiliaria colonial", "siic de paris"],
        "names_en": ["colonial"],
        "weight":   0.010,
        "sector":   "Real Estate",
    },
}

# ── Flat lookup: any keyword/ticker → (company_name, weight) ─────────────────
_CONSTITUENT_LOOKUP: dict[str, tuple[str, float]] = {}
for _name, _info in IBEX35_CONSTITUENTS.items():
    for _kw in _info["names_es"] + _info["names_en"] + _info["tickers"]:
        _CONSTITUENT_LOOKUP[_kw.lower()] = (_name, _info["weight"])


def classify_article(text: str) -> dict:
    """
    Classify an article by target type and compute relevance score.

    Scoring rationale (aligned with aspect-based sentiment literature):
      - Index mentions: highest base relevance (direct market impact)
      - Constituent: relevance proportional to index weight
      - Macro: bounded at 0.6 (indirect, diffuse effect)
      - Multiple keyword matches increase confidence (sum-based)

    Returns:
        target_type       : 'index' | 'constituent' | 'macro' | 'irrelevant'
        target_name       : company name (for constituent) or 'IBEX35' / None
        relevance_score   : float ∈ [0, 1]
        constituent_weight: index weight if constituent, else 0
    """
    text_lower = text.lower()

    # Index mentions — highest priority
    idx_hits = sum(1 for kw in INDEX_KEYWORDS if kw in text_lower)
    if idx_hits > 0:
        return {
            "target_type":        "index",
            "target_name":        "IBEX35",
            "relevance_score":    min(1.0, 0.60 + idx_hits * 0.08),
            "constituent_weight": 1.0,
        }

    # Constituent mentions — pick the highest-weight company found
    best_name, best_weight = None, 0.0
    for kw, (comp, weight) in _CONSTITUENT_LOOKUP.items():
        if kw in text_lower and weight > best_weight:
            best_name, best_weight = comp, weight

    if best_name is not None:
        return {
            "target_type":        "constituent",
            "target_name":        best_name,
            "relevance_score":    min(1.0, 0.40 + best_weight * 1.5),
            "constituent_weight": best_weight,
        }

    # Macro keywords
    macro_hits = sum(1 for kw in MACRO_KEYWORDS if kw in text_lower)
    if macro_hits > 0:
        return {
            "target_type":        "macro",
            "target_name":        None,
            "relevance_score":    min(0.60, 0.20 + macro_hits * 0.08),
            "constituent_weight": 0.0,
        }

    return {
        "target_type":        "irrelevant",
        "target_name":        None,
        "relevance_score":    0.0,
        "constituent_weight": 0.0,
    }
