# IBEX35 Forecasting System

A walk-forward machine learning system for predicting the direction of the IBEX35 index over 1-day and 5-day horizons, with full SHAP explainability and a local Streamlit dashboard.

---

## Dashboard

The dashboard is the main entry point for exploring signals, indicators, and model results.

### Launch

```bash
# From the project root
streamlit run app/dashboard.py
```

Then open **http://localhost:8501** in your browser.

### Tabs

| Tab | What you get |
|-----|-------------|
| **Señal** | Current UP/DOWN/NEUTRAL signal with probability bars and the top SHAP drivers explaining today's prediction |
| **Mercado** | Interactive candlestick chart with SMA/EMA overlays, Bollinger Bands, support/resistance levels, and volume |
| **Indicadores** | Color-coded state badges for all 4 technical categories (Tendencia · Momentum · Volatilidad · Volumen) and classical pivot points |
| **Explicabilidad** | On-demand SHAP computation: global feature importance bar chart + per-prediction waterfall |
| **Resultados** | Walk-forward ML metrics table (accuracy, AUC, F1) and a predictions timeline with probability vs correct/incorrect markers |
| **Backtest** | Equity curves (strategy vs buy-and-hold) and economic metrics: annual return, Sharpe ratio, max drawdown, Calmar |

> **Note:** The dashboard reads `results/latest_prediction.json`. If that file is missing or stale, run `python predict.py --no-multiasset --shap` first.

---

## Full Workflow

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

> **Python 3.10+** required. XGBoost is pinned to `<3.0` — do not upgrade, as XGBoost 3.x breaks SHAP's TreeExplainer.

### 2. Fetch data

Data is downloaded automatically from Yahoo Finance on first run and cached under `data/cache/`.

### 3. Train models

```bash
# Train all 4 models for 1-day direction
python train.py --model xgboost       --target target_dir_1d --no-multiasset
python train.py --model lgbm          --target target_dir_1d --no-multiasset
python train.py --model logistic      --target target_dir_1d --no-multiasset
python train.py --model random_forest --target target_dir_1d --no-multiasset

# Train XGBoost for 5-day direction
python train.py --model xgboost --target target_dir_5d --no-multiasset
```

Models are saved to `results/models/`. Walk-forward evaluation metrics are printed to the console and saved to `results/<target>_summary.json`.

### 4. Generate the latest prediction

```bash
python predict.py --no-multiasset --shap
```

Outputs `results/latest_prediction.json` with the signal, probability, and top SHAP drivers.

### 5. Launch the dashboard

```bash
streamlit run app/dashboard.py
```

---

## Project Structure

```
IBEX35_Prediction/
├── app/
│   ├── dashboard.py        # Streamlit dashboard (6 tabs)
│   └── api.py              # FastAPI REST endpoint (optional)
├── src/
│   ├── data/
│   │   ├── fetch.py        # Yahoo Finance downloader + parquet cache
│   │   ├── features.py     # 109 technical features across 5 categories
│   │   ├── multiasset.py   # S&P500, DAX, EUR/USD, VIX, Brent, US10y lags
│   │   └── sentiment.py    # RSS-based news sentiment (V2 stub)
│   ├── models/
│   │   ├── baseline.py     # XGBoost, LightGBM, LogReg, RF factories
│   │   ├── ensemble.py     # Stacking ensemble + model save/load with feature_names
│   │   └── lstm.py         # LSTM scaffold (V3)
│   ├── evaluation/
│   │   ├── backtest.py     # Walk-forward backtester with transaction costs
│   │   └── metrics.py      # Classification + economic metrics
│   ├── explainability/
│   │   └── shap_analysis.py # SHAP TreeExplainer / LinearExplainer + feature alignment
│   └── utils/
│       ├── config.py
│       └── logging.py      # UTF-8 stdout fix for Windows
├── train.py                # Walk-forward training pipeline
├── predict.py              # Latest-row prediction + SHAP drivers
├── config.yaml             # All hyperparameters and paths
└── requirements.txt
```

---

## Technical Indicators

The feature matrix includes **109 base features** across 5 categories:

| Category | Indicators |
|----------|-----------|
| **Tendencia** | SMA 20/50/200, EMA 8/21, ADX 14/21, DI+/DI−, price-MA crossovers, linear regression slope 10/20d |
| **Momentum** | RSI 14/21, Stochastic %K/%D, Williams %R, CCI 20, ROC 5/10/21d, MACD line/signal/histogram |
| **Volatilidad** | Bollinger Bands (width, %B), ATR 14/21 normalised, Keltner Channels, historical volatility, Chaikin Volatility |
| **Volumen** | OBV (normalised), CMF 20, MFI 14, ADL, PVT |
| **Pivotes** | Classical pivot points, support S1/S2, resistance R1/R2, distance to 52-week high/low |

---

## Models

| Model | Type | Notes |
|-------|------|-------|
| XGBoost | Gradient Boosting | Main model, early stopping, SHAP-compatible |
| LightGBM | Gradient Boosting | Faster alternative |
| Logistic Regression | Linear | Baseline with StandardScaler |
| Random Forest | Ensemble | 300 trees, balanced class weights |

Walk-forward evaluation uses an **expanding window** (no random shuffling): 36-month initial training, 3-month steps, 3-month validation folds. Transaction costs: **10 bps per trade**.

---

## Multi-Asset Context (optional)

When `--multiasset` is passed, 20 additional lagged features are added:

- Lag-1/2/5 returns of: S&P 500, DAX, EUR/USD, Brent
- VIX level + 20-day z-score
- US 10-year yield + lag-1 return

> Current trained models use `--no-multiasset` (109 features). To use multi-asset features, retrain without the flag.

---

## Roadmap

| Phase | Status |
|-------|--------|
| V1 — Core ML pipeline (XGBoost, LightGBM, walk-forward, backtest, SHAP, dashboard) | Done |
| V2 — News sentiment (FinBERT / RoBERTa-ES) | Stub ready |
| V3 — LSTM sequence model | Scaffold ready |
| V4 — Reinforcement Learning agent | Planned |
