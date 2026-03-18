"""
IBEX35 entity knowledge base for news relevance classification.

Provides keyword patterns to classify a news article as:
  - "index"       : directly about the IBEX35 or Spanish equity market
  - "constituent" : about a specific IBEX35 company
  - "macro"       : about macro/rates/ECB that affect Spanish equities
  - "irrelevant"  : no clear Spanish equity relevance

Also provides constituent weights for roll-up aggregation.
"""
from __future__ import annotations

# ── Direct IBEX35 / Spanish equity market ────────────────────────────────────
INDEX_KEYWORDS: list[str] = [
    "ibex", "ibex 35", "ibex35",
    "bolsa española", "bolsa de madrid", "bolsa madrid",
    "mercado español", "renta variable española",
    "bme", "bolsas y mercados",
    "bolsa española",
    "indice ibex",
    "benchmark español",
    "ibex continuo",
    "mercado continuo",
]

# ── Macro / systemic factors relevant to Spain ────────────────────────────────
MACRO_KEYWORDS: list[str] = [
    "bce", "banco central europeo", "ecb", "european central bank",
    "tipos de interés", "subida de tipos", "bajada de tipos", "tipo de interés",
    "inflación", "inflacion", "euribor",
    "prima de riesgo", "spread", "deuda española",
    "deuda soberana", "bono español",
    "pib españa", "economia española", "recesión españa",
    "eurozona", "euro area", "zona euro",
    "fed", "reserva federal",
    "geopolitic", "guerra comercial", "aranceles",
    "volatilidad europea", "vstoxx", "vix",
    "euro stoxx", "stoxx 50",
    "esm", "mede", "mecanismo europeo",
]

# ── IBEX35 constituents — names, tickers, and approximate index weights ───────
# Weights are approximate (2024 composition) and used for roll-up aggregation.
IBEX35_CONSTITUENTS: dict[str, dict] = {
    "Inditex": {
        "tickers":  ["ITX", "ITX.MC"],
        "names_es": ["inditex", "zara", "mango", "bershka"],  # major brands
        "names_en": ["inditex", "zara"],
        "weight":   0.155,
        "sector":   "Consumer",
    },
    "BBVA": {
        "tickers":  ["BBVA", "BBVA.MC"],
        "names_es": ["bbva", "banco bilbao vizcaya", "bbva bancomer"],
        "names_en": ["bbva"],
        "weight":   0.115,
        "sector":   "Financials",
    },
    "Santander": {
        "tickers":  ["SAN", "SAN.MC"],
        "names_es": ["santander", "banco santander", "grupo santander"],
        "names_en": ["santander"],
        "weight":   0.115,
        "sector":   "Financials",
    },
    "Iberdrola": {
        "tickers":  ["IBE", "IBE.MC"],
        "names_es": ["iberdrola"],
        "names_en": ["iberdrola"],
        "weight":   0.095,
        "sector":   "Utilities",
    },
    "Ferrovial": {
        "tickers":  ["FER", "FER.MC"],
        "names_es": ["ferrovial"],
        "names_en": ["ferrovial"],
        "weight":   0.055,
        "sector":   "Infrastructure",
    },
    "Amadeus": {
        "tickers":  ["AMS", "AMS.MC"],
        "names_es": ["amadeus"],
        "names_en": ["amadeus"],
        "weight":   0.050,
        "sector":   "Technology",
    },
    "CaixaBank": {
        "tickers":  ["CABK", "CABK.MC"],
        "names_es": ["caixabank", "la caixa", "caixa bank"],
        "names_en": ["caixabank"],
        "weight":   0.042,
        "sector":   "Financials",
    },
    "Repsol": {
        "tickers":  ["REP", "REP.MC"],
        "names_es": ["repsol"],
        "names_en": ["repsol"],
        "weight":   0.040,
        "sector":   "Energy",
    },
    "Telefonica": {
        "tickers":  ["TEF", "TEF.MC"],
        "names_es": ["telefonica", "telefónica", "movistar"],
        "names_en": ["telefonica", "movistar"],
        "weight":   0.038,
        "sector":   "Telecom",
    },
    "Cellnex": {
        "tickers":  ["CLNX", "CLNX.MC"],
        "names_es": ["cellnex"],
        "names_en": ["cellnex"],
        "weight":   0.030,
        "sector":   "Infrastructure",
    },
    "Aena": {
        "tickers":  ["AENA", "AENA.MC"],
        "names_es": ["aena"],
        "names_en": ["aena"],
        "weight":   0.030,
        "sector":   "Transport",
    },
    "Naturgy": {
        "tickers":  ["NTGY", "NTGY.MC"],
        "names_es": ["naturgy", "gas natural"],
        "names_en": ["naturgy"],
        "weight":   0.028,
        "sector":   "Utilities",
    },
    "ACS": {
        "tickers":  ["ACS", "ACS.MC"],
        "names_es": ["acs", "grupo acs"],
        "names_en": ["acs"],
        "weight":   0.025,
        "sector":   "Construction",
    },
    "Endesa": {
        "tickers":  ["ELE", "ELE.MC"],
        "names_es": ["endesa"],
        "names_en": ["endesa"],
        "weight":   0.022,
        "sector":   "Utilities",
    },
    "Sabadell": {
        "tickers":  ["SAB", "SAB.MC"],
        "names_es": ["sabadell", "banco sabadell"],
        "names_en": ["sabadell"],
        "weight":   0.018,
        "sector":   "Financials",
    },
}

# Build a flat lookup: any keyword → (company_name, weight)
_CONSTITUENT_LOOKUP: dict[str, tuple[str, float]] = {}
for _name, _info in IBEX35_CONSTITUENTS.items():
    for _kw in _info["names_es"] + _info["names_en"] + _info["tickers"]:
        _CONSTITUENT_LOOKUP[_kw.lower()] = (_name, _info["weight"])


def classify_article(text: str) -> dict:
    """
    Classify an article by target type and relevance score.

    Returns:
        target_type    : 'index' | 'constituent' | 'macro' | 'irrelevant'
        target_name    : company name (for constituent) or None
        relevance_score: float in [0, 1]
        constituent_weight: index weight if constituent, else 0
    """
    text_lower = text.lower()

    # Check direct index mention first (highest priority)
    idx_matches = sum(1 for kw in INDEX_KEYWORDS if kw in text_lower)
    if idx_matches > 0:
        return {
            "target_type":         "index",
            "target_name":         "IBEX35",
            "relevance_score":     min(1.0, 0.6 + idx_matches * 0.1),
            "constituent_weight":  1.0,
        }

    # Check constituent mentions
    found_constituent: str | None = None
    found_weight = 0.0
    for kw, (comp, weight) in _CONSTITUENT_LOOKUP.items():
        if kw in text_lower:
            if weight > found_weight:   # take highest-weight constituent
                found_constituent = comp
                found_weight      = weight

    if found_constituent:
        return {
            "target_type":         "constituent",
            "target_name":         found_constituent,
            "relevance_score":     min(1.0, 0.4 + found_weight * 2),
            "constituent_weight":  found_weight,
        }

    # Check macro keywords
    macro_matches = sum(1 for kw in MACRO_KEYWORDS if kw in text_lower)
    if macro_matches > 0:
        return {
            "target_type":         "macro",
            "target_name":         None,
            "relevance_score":     min(0.6, 0.2 + macro_matches * 0.1),
            "constituent_weight":  0.0,
        }

    return {
        "target_type":         "irrelevant",
        "target_name":         None,
        "relevance_score":     0.0,
        "constituent_weight":  0.0,
    }
