# IBEX35 Forecasting System

A walk-forward machine learning system for predicting the direction of the IBEX35 index over 1-day and 5-day horizons, with SHAP explainability, a local Streamlit dashboard, and a REST API.

> **Data integrity**: all features are strictly lagged (no lookahead); vol-regime labels use expanding quantiles; walk-forward splits are time-ordered with no shuffling.

---

## Quick start

```bash
pip install -r requirements.txt

# 1. Train all models + auto-select champion (data downloaded automatically)
python train.py --all-targets
# Trains XGBoost, LightGBM, LogReg, RF, Ensemble for each target.
# Compares by walk-forward AUC + Sharpe. Saves best to results/champion.json.

# 2. Generate latest signal + SHAP drivers  (uses champion model automatically)
python predict.py --shap

# 3. Launch dashboard
streamlit run app/dashboard.py        # http://localhost:8501

# 4. (optional) Start REST API
uvicorn app.api:app --reload --port 8000  # http://localhost:8000/docs
```

To retrain a specific model or target:
```bash
python train.py --model xgboost --target target_dir_1d
python train.py --model ensemble
python predict.py --model xgboost   # override champion for this run
```

---

## Dashboard

Open **http://localhost:8501** after running `streamlit run app/dashboard.py`.

| Tab | Contents |
|-----|----------|
| **📝 Summary** | **Operative summary**: plain-English narrative explaining *why* the model makes its call — top SHAP drivers, three-layer sentiment state, recent IBEX-relevant headlines with per-article scores |
| **📊 Signal** | UP/DOWN/NEUTRAL signal with probability bars + top SHAP drivers |
| **📈 Market** | Interactive candlestick with SMA/Bollinger/support-resistance overlays and volume |
| **🔧 Indicators** | Color-coded state badges for Trend · Momentum · Volatility · Volume · Pivots |
| **🧠 Explainability** | On-demand SHAP: global feature importance + per-prediction waterfall |
| **📋 Results** | Walk-forward ML metrics (accuracy, AUC, F1) + predictions timeline |
| **💰 Backtest** | Equity curves (strategy vs buy-and-hold) with adjustable confidence threshold and costs |

The sidebar has a **Refresh data** button and usage instructions. The signal tab reads `results/latest_prediction.json`; run `python predict.py --shap` to update it.

---

## REST API

```bash
uvicorn app.api:app --reload --port 8000
```

Open **http://localhost:8000/docs** for the interactive Swagger UI.

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Liveness check |
| GET | `/signal` | Latest cached market signal |
| GET | `/summary?live_news=true` | **Operative summary**: prediction narrative + sentiment + media |
| GET | `/indicators` | All technical indicator states |
| GET | `/latest` | Most recent feature vector |
| GET | `/history?n=252` | Last N days of OHLCV + log-return |
| GET | `/results/{target}` | Walk-forward summary for a trained target |
| POST | `/predict` | Run fresh prediction (fast, ~2s) |
| POST | `/predict/shap` | Run fresh prediction with SHAP drivers (~30s) |

---

## Research Notebook

```bash
jupyter lab notebooks/ibex_study.ipynb
```

The notebook answers six literature-driven research questions about what drives IBEX35 forecastability:

| # | Question | Key finding |
|---|----------|-------------|
| Q1 | Does EURO STOXX 50 add incremental predictive power? | VIX level has the highest IC; STOXX50 lags add small signal |
| Q2 | Do volatility-regime variables matter more than trend variables? | High-vol regime shows wider distributions; momentum degrades |
| Q3 | Does constituent correlation structure carry signal? | Average pairwise correlation spikes in crises (2008, 2020) |
| Q4 | Which feature family drives the model most? | Trend > Lagged Returns > Volatility (XGBoost, 1d direction) |
| Q5 | Is direction prediction easier than return regression? | Yes — AUC > 0.5 achievable; return R² ≈ 0 |
| Q6 | Are drivers stable across walk-forward windows? | AUC fluctuates; high-vol regimes are hardest to predict |

Saved plots: `results/study_0*.png`

**Literature basis**: Giantsidi & Tarantola (2025) deep-learning review; 2025 indicator study (momentum/trend/volatility/volume families); STOXX white paper (VSTOXX–EURO STOXX correlation); Finance Research Letters 2024 (constituent correlation).

---

## Project structure

```
IBEX35_Prediction/
├── app/
│   ├── dashboard.py        # Streamlit dashboard (6 tabs)
│   └── api.py              # FastAPI REST API
├── notebooks/
│   └── ibex_study.ipynb    # Research notebook (6 questions)
├── src/
│   ├── data/
│   │   ├── fetch.py        # Yahoo Finance downloader + parquet cache
│   │   ├── features.py     # 109 technical features, 5 categories (no lookahead)
│   │   ├── multiasset.py   # EURO STOXX 50, DAX, S&P500, EUR/USD, VIX, Brent, US10y
│   │   └── sentiment.py    # RSS-based news sentiment (V2 stub)
│   ├── models/
│   │   ├── baseline.py     # XGBoost, LightGBM, LogReg, RF factories
│   │   ├── ensemble.py     # Stacking + model save/load with feature_names bundle
│   │   └── lstm.py         # LSTM scaffold (V3)
│   ├── evaluation/
│   │   ├── backtest.py     # Walk-forward (expanding or rolling window) + backtest
│   │   └── metrics.py      # Classification + economic metrics
│   └── explainability/
│       └── shap_analysis.py # SHAP with feature-count alignment fix
├── train.py                # Walk-forward training pipeline
├── predict.py              # Latest-row prediction + SHAP
└── config.yaml             # All hyperparameters, paths, and walk-forward config
```

---

## Technical indicators — 109 base features

| Category | Indicators |
|----------|-----------|
| **Trend** | SMA 5/10/20/50/100/200 distance, EMA 8/21/55 distance, ADX 14/21, DI+/DI−, 3 SMA crossovers, LinReg slope 10/20/50d, 52-week high/low distance |
| **Momentum** | RSI 7/14/21, Stochastic %K/%D, Williams %R 14, CCI 14/20, ROC 5/10/21d, MACD line/signal/histogram |
| **Volatility** | Bollinger %B/width/band distances, ATR 7/14/21 (normalised), ATR ratio, Keltner Channel %, HV 5/10/21/63d, HV ratios, Chaikin Volatility |
| **Volume** | OBV slope, CMF 20, MFI 14, ADL slope, PVT slope, volume z-score, volume ratios, signed price×volume |
| **Pivots** | Rolling resistance/support distances (10/20/50d), classical pivot + R1/S1/R2/S2, round-level distance |
| **Base** | Lagged returns (1/2/3/5/10/21d), rolling mean/std/skew (5/10/21/63d), HL ratio, calendar features, realized volatility + slope |

---

## Multi-asset features (optional)

| Asset | Ticker | Features |
|-------|--------|----------|
| EURO STOXX 50 | ^STOXX50E | Lag-1/2/5 returns + relative strength vs IBEX |
| DAX | ^GDAXI | Lag-1/2/5 returns |
| S&P 500 | ^GSPC | Lag-1/2/5 returns |
| EUR/USD | EURUSD=X | Lag-1/2/5 returns |
| VIX | ^VIX | Lag-1/2/5 returns + level + 21d z-score |
| VSTOXX | ^V2TX | Lag-1/2/5 returns + level + 21d z-score *(unavailable on Yahoo Finance)* |
| Brent Oil | BZ=F | Lag-1/2/5 returns |
| US 10y yield | ^TNX | Lag-1/2/5 returns |

Train with multi-asset: remove `--no-multiasset` flag.

---

## Walk-forward design

| Parameter | Default | Notes |
|-----------|---------|-------|
| Initial train window | 36 months | Expanding from first observation |
| Step | 3 months | New validation fold every quarter |
| Validation window | 3 months | Out-of-sample per fold |
| Max train months | null (expanding) | Set to e.g. `48` for rolling 4-year window |
| Transaction cost | 10 bps | One-way, applied on each position change |
| Confidence threshold | 55% | P(up) ≥ 55% to go long |

**On COVID (2020)**: the system includes 2020 data by design. Recent literature (2022–2025) recommends keeping extreme-regime periods in training — they teach the model real tail-risk behavior. A rolling window (`max_train_months: 48`) naturally down-weights pre-crisis data without discarding it.

---

## Dependencies

| Package | Notes |
|---------|-------|
| `xgboost>=2.1.0,<3.0.0` | **Pinned** — XGBoost 3.x breaks SHAP's TreeExplainer |
| `shap>=0.45.0` | Required for explainability tabs |
| `streamlit>=1.35.0` | Dashboard |
| `fastapi>=0.110.0` + `uvicorn` | REST API |

```bash
pip install -r requirements.txt
```

---

## Sentiment pipeline

Three-layer composite aligned with Tetlock (2007) / Loughran & McDonald (2011):

| Layer | Weight | Description |
|-------|--------|-------------|
| Direct IBEX | α = 0.50 | Articles mentioning IBEX35 / bolsa española |
| Constituent roll-up | β = 0.30 | Company news weighted by index weight |
| Macro EU/Spain | γ = 0.20 | ECB, rates, eurozone articles |

**Per-article weight:** `w_i = w_rel · w_src · w_time · w_novelty`
- `w_rel`: relevance score from entity classification
- `w_src`: source credibility (Reuters=1.0, Expansión=0.85, general press=0.65)
- `w_time`: exponential decay with 18h half-life (Da et al. 2011)
- `w_novelty`: 1/√rank within same topic per day (Chan 2003)

**Two-track design** (coverage integrity principle):

| Track | Config | Period | Purpose |
|-------|--------|--------|---------|
| A | `include_sentiment: false` | Full history 2007–today | Price/vol/regime baseline |
| B | `include_sentiment: true` | `sentiment_start_date` onwards | Base vs base+sentiment ablation |

Dates before `sentiment.start_date` are `NaN` (archive unavailable), not `0` (no news).
Only run Track A vs Track B on the **same date range** to measure incremental value.

Set `sentiment.start_date` to the earliest date with trustworthy, stable RSS coverage.

---

## Roadmap

| Phase | Status | Description |
|-------|--------|-------------|
| V1 | Done | XGBoost · LightGBM · LogReg · RF · walk-forward · SHAP · dashboard · API |
| V2 | Done | Three-layer sentiment (keyword fallback + optional transformers) · operative summary |
| V3 | Scaffold | Probabilistic / quantile forecasting + LSTM |
| V4 | Planned | Reinforcement learning agent (ABIDES-style) |
