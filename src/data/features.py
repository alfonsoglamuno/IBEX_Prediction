"""
Feature engineering for IBEX35 forecasting.

Indicator categories covered:
  1. TREND        — SMA, EMA, crossovers, ADX, trend strength
  2. MOMENTUM     — RSI (7/14), Stochastic, CCI, MACD
  3. VOLATILITY   — Bollinger Bands, ATR, Keltner Channels, HV
  4. VOLUME       — OBV, CMF, MFI, A/D Line, PVT
  5. LOCAL PIVOTS — Support/Resistance, swing highs/lows, pivot points
  6. LAGGED returns, rolling stats, calendar, volatility regime
  7. TARGETS      — direction 1d/5d, log-return 5d, vol regime bucket
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import argrelextrema

from src.utils.config import get
from src.utils.logging import get_logger

log = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Primitive helpers
# ─────────────────────────────────────────────────────────────────────────────

def _log_return(s: pd.Series, n: int = 1) -> pd.Series:
    return np.log(s / s.shift(n))


def _wilder_ema(s: pd.Series, period: int) -> pd.Series:
    """Wilder's smoothing (used by ATR, ADX, RSI)."""
    return s.ewm(alpha=1.0 / period, adjust=False).mean()


# ── Trend helpers ─────────────────────────────────────────────────────────────

def _adx(high: pd.Series, low: pd.Series, close: pd.Series,
         period: int = 14) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Returns (ADX, +DI, -DI) using Wilder's smoothing."""
    up   = high.diff()
    dn   = -low.diff()
    pdm  = up.where((up > dn) & (up > 0), 0.0)
    ndm  = dn.where((dn > up) & (dn > 0), 0.0)

    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs(),
    ], axis=1).max(axis=1)

    atr_w = _wilder_ema(tr, period)
    pdi   = 100 * _wilder_ema(pdm, period) / (atr_w + 1e-9)
    ndi   = 100 * _wilder_ema(ndm, period) / (atr_w + 1e-9)
    dx    = 100 * (pdi - ndi).abs() / (pdi + ndi + 1e-9)
    adx   = _wilder_ema(dx, period)
    return adx, pdi, ndi


def _linreg_slope(s: pd.Series, window: int) -> pd.Series:
    """Rolling linear-regression slope (normalized by mean price)."""
    x = np.arange(window, dtype=float)
    x -= x.mean()
    def _slope(y):
        return np.dot(x, y) / (np.dot(x, x) + 1e-9)
    return s.rolling(window).apply(_slope, raw=True) / (s.rolling(window).mean() + 1e-9)


# ── Momentum helpers ──────────────────────────────────────────────────────────

def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain  = _wilder_ema(delta.clip(lower=0), period)
    loss  = _wilder_ema((-delta).clip(lower=0), period)
    rs    = gain / (loss + 1e-9)
    return 100 - 100 / (1 + rs)


def _stochastic(high: pd.Series, low: pd.Series, close: pd.Series,
                k: int = 14, d: int = 3) -> tuple[pd.Series, pd.Series]:
    lo_k  = low.rolling(k).min()
    hi_k  = high.rolling(k).max()
    pct_k = 100 * (close - lo_k) / (hi_k - lo_k + 1e-9)
    pct_d = pct_k.rolling(d).mean()
    return pct_k, pct_d


def _williams_r(high: pd.Series, low: pd.Series, close: pd.Series,
                period: int = 14) -> pd.Series:
    hi_p = high.rolling(period).max()
    lo_p = low.rolling(period).min()
    return -100 * (hi_p - close) / (hi_p - lo_p + 1e-9)


def _cci(high: pd.Series, low: pd.Series, close: pd.Series,
         period: int = 20) -> pd.Series:
    tp    = (high + low + close) / 3
    sma   = tp.rolling(period).mean()
    mad   = tp.rolling(period).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    return (tp - sma) / (0.015 * mad + 1e-9)


# ── Volatility helpers ────────────────────────────────────────────────────────

def _atr(high: pd.Series, low: pd.Series, close: pd.Series,
         period: int = 14) -> pd.Series:
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs(),
    ], axis=1).max(axis=1)
    return _wilder_ema(tr, period)


def _keltner_pct(high: pd.Series, low: pd.Series, close: pd.Series,
                 ema_period: int = 20, atr_period: int = 10,
                 mult: float = 2.0) -> pd.Series:
    ema   = close.ewm(span=ema_period, adjust=False).mean()
    atr_v = _atr(high, low, close, atr_period)
    upper = ema + mult * atr_v
    lower = ema - mult * atr_v
    return (close - lower) / (upper - lower + 1e-9)


# ── Volume helpers ────────────────────────────────────────────────────────────

def _obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = np.sign(close.diff()).fillna(0)
    return (direction * volume).cumsum()


def _cmf(high: pd.Series, low: pd.Series, close: pd.Series,
         volume: pd.Series, period: int = 20) -> pd.Series:
    mfm = ((close - low) - (high - close)) / (high - low + 1e-9)
    return (mfm * volume).rolling(period).sum() / (volume.rolling(period).sum() + 1e-9)


def _mfi(high: pd.Series, low: pd.Series, close: pd.Series,
         volume: pd.Series, period: int = 14) -> pd.Series:
    tp   = (high + low + close) / 3
    mf   = tp * volume
    up   = tp > tp.shift(1)
    pmf  = mf.where(up,  0.0).rolling(period).sum()
    nmf  = mf.where(~up, 0.0).rolling(period).sum()
    mfr  = pmf / (nmf + 1e-9)
    return 100 - 100 / (1 + mfr)


def _adl(high: pd.Series, low: pd.Series, close: pd.Series,
         volume: pd.Series) -> pd.Series:
    clv = ((close - low) - (high - close)) / (high - low + 1e-9)
    return (clv * volume).cumsum()


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Input:  raw OHLCV DataFrame (DatetimeIndex, columns: Open High Low Close Volume)
    Output: feature matrix + target columns. NaN rows (feature side) trimmed.

    All columns are accumulated in a dict first, then assembled in a single
    pd.DataFrame call to avoid the PerformanceWarning from repeated inserts.
    """
    cfg_lags = get("features.lags",    [1, 2, 3, 5, 10, 21])
    cfg_wins = get("features.windows", [5, 10, 21, 63])
    use_vol  = get("features.include_volume", True)

    cols: dict[str, pd.Series] = {}   # accumulator — no fragmentation

    c  = df["Close"]
    h  = df["High"]
    lo = df["Low"]
    v  = df["Volume"] if "Volume" in df.columns else pd.Series(np.nan, index=df.index)

    daily_lr = _log_return(c, 1)

    # ── 1. Lagged log returns ─────────────────────────────────────────────────
    for lag in cfg_lags:
        cols[f"ret_lag{lag}"] = daily_lr.shift(lag)

    # ── 2. Rolling statistics ─────────────────────────────────────────────────
    # Note: ret_std_*d removed — identical to hv_*d / sqrt(252), fully redundant.
    for w in cfg_wins:
        roll = daily_lr.rolling(w)
        cols[f"ret_mean_{w}d"] = roll.mean()
        cols[f"ret_skew_{w}d"] = roll.skew()
        cols[f"hl_ratio_{w}d"] = (h.rolling(w).max() - lo.rolling(w).min()) / (c.rolling(w).mean() + 1e-9)

    # ═══════════════════════════════════════════════════════════════════════════
    # CATEGORY 1 — TREND indicators
    # ═══════════════════════════════════════════════════════════════════════════

    # SMA 5/10 removed: highly correlated with ret_mean_5d/10d (r > 0.97)
    for p in [20, 50, 100, 200]:
        sma = c.rolling(p).mean()
        cols[f"sma{p}_dist"] = (c - sma) / (sma + 1e-9)

    for p in [8, 55]:
        ema = c.ewm(span=p, adjust=False).mean()
        cols[f"ema{p}_dist"] = (c - ema) / (ema + 1e-9)

    sma5   = c.rolling(5).mean()
    sma20  = c.rolling(20).mean()
    sma50  = c.rolling(50).mean()
    sma200 = c.rolling(200).mean()
    cols["cross_5_20"]   = (sma5  - sma20)  / (sma20  + 1e-9)
    cols["cross_20_50"]  = (sma20 - sma50)  / (sma50  + 1e-9)
    cols["cross_50_200"] = (sma50 - sma200) / (sma200 + 1e-9)

    adx14, pdi14, ndi14 = _adx(h, lo, c, 14)
    adx21, pdi21, ndi21 = _adx(h, lo, c, 21)
    cols["adx14"]     = adx14
    cols["di_diff14"] = (pdi14 - ndi14) / 100.0
    cols["adx21"]     = adx21

    for w in [10, 20, 50]:
        cols[f"lr_slope_{w}d"] = _linreg_slope(c, w)

    for w in [63, 126, 252]:
        cols[f"dist_52w_high_{w}d"] = c / (c.rolling(w).max() + 1e-9) - 1
        cols[f"dist_52w_low_{w}d"]  = c / (c.rolling(w).min() + 1e-9) - 1

    # ═══════════════════════════════════════════════════════════════════════════
    # CATEGORY 2 — MOMENTUM indicators
    # ═══════════════════════════════════════════════════════════════════════════

    for p in [7, 14]:
        cols[f"rsi_{p}"] = _rsi(c, p)
    cols["rsi14_slope5"] = _rsi(c, 14).diff(5)

    stoch_k, stoch_d = _stochastic(h, lo, c, 14, 3)
    cols["stoch_k"]    = stoch_k
    cols["stoch_d"]    = stoch_d
    cols["stoch_diff"] = stoch_k - stoch_d

    cols["cci20"] = _cci(h, lo, c, 20)

    ema12      = c.ewm(span=12, adjust=False).mean()
    ema26      = c.ewm(span=26, adjust=False).mean()
    macd_line  = ema12 - ema26
    macd_sig   = macd_line.ewm(span=9, adjust=False).mean()
    macd_hist  = (macd_line - macd_sig) / (c + 1e-9)
    cols["macd"]             = macd_line / (c + 1e-9)
    cols["macd_signal"]      = macd_sig  / (c + 1e-9)
    cols["macd_hist"]        = macd_hist
    cols["macd_hist_slope3"] = macd_hist.diff(3)

    # ═══════════════════════════════════════════════════════════════════════════
    # CATEGORY 3 — VOLATILITY indicators
    # ═══════════════════════════════════════════════════════════════════════════

    boll_mid = c.rolling(20).mean()
    boll_std = c.rolling(20).std()
    cols["boll_pct_b"]      = (c - (boll_mid - 2*boll_std)) / (4*boll_std + 1e-9)
    cols["boll_width"]      = (4*boll_std) / (boll_mid + 1e-9)
    cols["boll_upper_dist"] = (c - (boll_mid + 2*boll_std)) / (c + 1e-9)
    cols["boll_lower_dist"] = (c - (boll_mid - 2*boll_std)) / (c + 1e-9)

    for p in [7, 14]:
        cols[f"atr_{p}_norm"] = _atr(h, lo, c, p) / (c + 1e-9)
    cols["atr_ratio_7_21"] = _atr(h, lo, c, 7) / (_atr(h, lo, c, 21) + 1e-9)
    cols["keltner_pct"]    = _keltner_pct(h, lo, c, 20, 10, 2.0)

    hv = {w: daily_lr.rolling(w).std() * np.sqrt(252) for w in [5, 10, 21, 63]}
    for w, s in hv.items():
        cols[f"hv_{w}d"] = s
    cols["hv_ratio_5_21"]  = hv[5]  / (hv[21]  + 1e-9)
    cols["hv_ratio_21_63"] = hv[21] / (hv[63]  + 1e-9)

    hl_ema = (h - lo).ewm(span=10, adjust=False).mean()
    cols["chaikin_vol"] = hl_ema.pct_change(10)

    # ═══════════════════════════════════════════════════════════════════════════
    # CATEGORY 4 — VOLUME indicators
    # ═══════════════════════════════════════════════════════════════════════════

    if use_vol and v.notna().sum() > 50:
        v_mean21 = v.rolling(21).mean()
        v_std21  = v.rolling(21).std()
        v_z21    = (v - v_mean21) / (v_std21 + 1e-9)
        cols["vol_z21"]         = v_z21
        cols["vol_ratio_5_21"]  = v.rolling(5).mean() / (v_mean21 + 1e-9)
        cols["vol_ratio_1_21"]  = v / (v_mean21 + 1e-9)
        cols["price_vol_signed"]= daily_lr * v_z21

        obv = _obv(c, v)
        cols["obv_slope5"]  = obv.diff(5)  / (v_mean21 + 1e-9)
        cols["obv_slope21"] = obv.diff(21) / (v_mean21 + 1e-9)
        cols["cmf20"]       = _cmf(h, lo, c, v, 20)
        cols["mfi14"]       = _mfi(h, lo, c, v, 14)

        adl = _adl(h, lo, c, v)
        cols["adl_slope5"]  = adl.diff(5)  / (v_mean21 + 1e-9)
        cols["adl_slope21"] = adl.diff(21) / (v_mean21 + 1e-9)

        pvt = (daily_lr * v).cumsum()
        cols["pvt_slope10"] = pvt.diff(10) / (v_mean21 + 1e-9)

    # ═══════════════════════════════════════════════════════════════════════════
    # CATEGORY 5 — LOCAL MAXIMA / MINIMA (Support & Resistance)
    # ═══════════════════════════════════════════════════════════════════════════

    for w in [10, 20, 50]:
        cols[f"dist_resistance_{w}d"] = (c - h.rolling(w).max().shift(1))  / (c + 1e-9)
        cols[f"dist_support_{w}d"]    = (c - lo.rolling(w).min().shift(1)) / (c + 1e-9)

    ph    = h.shift(1)
    pl    = lo.shift(1)
    pc    = c.shift(1)
    pivot = (ph + pl + pc) / 3
    r1    = 2*pivot - pl
    s1    = 2*pivot - ph
    r2    = pivot + (ph - pl)
    s2    = pivot - (ph - pl)
    cols["dist_pivot"] = (c - pivot) / (c + 1e-9)
    cols["dist_r1"]    = (c - r1)    / (c + 1e-9)
    cols["dist_s1"]    = (c - s1)    / (c + 1e-9)
    cols["dist_r2"]    = (c - r2)    / (c + 1e-9)
    cols["dist_s2"]    = (c - s2)    / (c + 1e-9)

    # Round-number level distance — magnitude from expanding median (no lookahead)
    exp_median = c.expanding(252).median().shift(1).ffill()
    magnitude  = (10 ** np.floor(np.log10(exp_median.clip(lower=1)))).fillna(1000)
    cols["dist_round_level"] = (c - (c / magnitude).round() * magnitude) / (c + 1e-9)

    # Calendar features removed (aggregate XGBoost importance < 0.03; day-of-week
    # and monthly effects are largely arbitraged away in modern equity markets).

    # ── 6. Volatility regime context ─────────────────────────────────────────
    # rv_21 == hv_21d (identical formula); use hv[21] to avoid the duplicate column.
    cols["rv_slope10"] = hv[21].diff(10)

    # ═══════════════════════════════════════════════════════════════════════════
    # TARGET VARIABLES  (forward-looking — use ONLY as y, never as X)
    # ═══════════════════════════════════════════════════════════════════════════
    fwd_1d = daily_lr.shift(-1)
    fwd_5d = _log_return(c, 5).shift(-5)

    cols["target_dir_1d"]    = (fwd_1d > 0).astype(int)
    cols["target_dir_5d"]    = (fwd_5d > 0).astype(int)
    cols["target_ret_1d"]    = fwd_1d
    cols["target_ret_5d"]    = fwd_5d

    # Vol-regime labels use expanding quantiles (no lookahead into future volatility)
    rv_q33 = hv[21].expanding(252).quantile(0.33).shift(1).ffill()
    rv_q66 = hv[21].expanding(252).quantile(0.66).shift(1).ffill()
    vol_regime = np.where(hv[21] < rv_q33, 0, np.where(hv[21] < rv_q66, 1, 2))
    cols["target_vol_regime"] = pd.Series(vol_regime, index=df.index, dtype=float)

    # ── Assemble DataFrame in one call (no fragmentation) ────────────────────
    out = pd.DataFrame(cols, index=df.index)

    # ── Optional sentiment layer ──────────────────────────────────────────────
    if get("features.include_sentiment", False):
        from src.data.sentiment_features import build_sentiment_features
        sent_df = build_sentiment_features(out.index)
        if not sent_df.empty:
            for col in sent_df.columns:
                out[col] = sent_df[col]

    feat_cols_local = [col for col in out.columns if not col.startswith("target_")]
    before = len(out)
    out.dropna(subset=feat_cols_local, inplace=True)
    log.info(f"Features: {len(feat_cols_local)} cols | {len(out)} rows (dropped {before - len(out)} NaN)")

    return out


# ─────────────────────────────────────────────────────────────────────────────
# Column selectors
# ─────────────────────────────────────────────────────────────────────────────

def feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if not c.startswith("target_")]


def target_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("target_")]


# ─────────────────────────────────────────────────────────────────────────────
# Indicator state (used by dashboard — returns human-readable status)
# ─────────────────────────────────────────────────────────────────────────────

def get_indicator_states(df_raw: pd.DataFrame) -> list[dict]:
    """
    Compute the current state of each technical indicator for the dashboard.
    Returns a list of dicts with keys:
        category, name, value, state (bullish/bearish/neutral/overbought/oversold), label
    """
    if len(df_raw) < 210:
        return []

    c  = df_raw["Close"]
    h  = df_raw["High"]
    lo = df_raw["Low"]
    v  = df_raw["Volume"] if "Volume" in df_raw.columns else None

    states = []

    def _s(category, name, value, state, label):
        states.append({
            "category": category,
            "name":     name,
            "value":    round(float(value), 4) if pd.notna(value) else None,
            "state":    state,
            "label":    label,
        })

    # ── TREND ─────────────────────────────────────────────────────────────────
    sma20_v  = c.rolling(20).mean().iloc[-1]
    sma50_v  = c.rolling(50).mean().iloc[-1]
    sma200_v = c.rolling(200).mean().iloc[-1]
    last_c   = c.iloc[-1]

    trend_20_50 = "bullish" if sma20_v > sma50_v else "bearish"
    _s("Trend", "SMA20 vs SMA50", (sma20_v - sma50_v) / sma50_v * 100,
       trend_20_50, f"SMA20={'above' if trend_20_50 == 'bullish' else 'below'} SMA50")

    trend_200 = "bullish" if last_c > sma200_v else "bearish"
    _s("Trend", "Price vs SMA200", (last_c - sma200_v) / sma200_v * 100,
       trend_200, f"Price {'above' if trend_200 == 'bullish' else 'below'} SMA200")

    adx_v, pdi_v, ndi_v = _adx(h, lo, c, 14)
    adx_last  = adx_v.iloc[-1]
    di_last   = pdi_v.iloc[-1] - ndi_v.iloc[-1]
    adx_state = "bullish" if adx_last > 25 and di_last > 0 else \
                "bearish" if adx_last > 25 and di_last < 0 else "neutral"
    _s("Trend", "ADX(14)", adx_last, adx_state,
       f"ADX={adx_last:.1f} ({'Strong trend' if adx_last > 25 else 'No trend'})")

    lr_slope = _linreg_slope(c, 20).iloc[-1] * 100
    _s("Trend", "LinReg Slope 20d", lr_slope,
       "bullish" if lr_slope > 0 else "bearish",
       f"20d linear regression: {lr_slope:+.2f}%/day")

    # ── MOMENTUM ──────────────────────────────────────────────────────────────
    rsi14 = _rsi(c, 14).iloc[-1]
    rsi_state = "overbought" if rsi14 > 70 else "oversold" if rsi14 < 30 else "neutral"
    _s("Momentum", "RSI(14)", rsi14, rsi_state,
       f"RSI={rsi14:.1f} ({'Overbought>70' if rsi14 > 70 else 'Oversold<30' if rsi14 < 30 else 'Neutral'})")

    stk, std = _stochastic(h, lo, c, 14, 3)
    stk_v = stk.iloc[-1]
    std_v = std.iloc[-1]
    stoch_state = "overbought" if stk_v > 80 else "oversold" if stk_v < 20 else \
                  "bullish" if stk_v > std_v else "bearish"
    _s("Momentum", "Stochastic(14,3)", stk_v, stoch_state,
       f"%K={stk_v:.1f}  %D={std_v:.1f}")

    ema12_v  = c.ewm(span=12, adjust=False).mean().iloc[-1]
    ema26_v  = c.ewm(span=26, adjust=False).mean().iloc[-1]
    macd_v   = ema12_v - ema26_v
    macd_sig_v = c.ewm(span=12, adjust=False).mean().sub(
                 c.ewm(span=26, adjust=False).mean()
                 ).ewm(span=9, adjust=False).mean().iloc[-1]
    hist_v   = macd_v - macd_sig_v
    macd_state = "bullish" if hist_v > 0 else "bearish"
    _s("Momentum", "MACD Histogram", hist_v / last_c * 100, macd_state,
       f"Histogram {'positive' if hist_v > 0 else 'negative'}")

    cci20 = _cci(h, lo, c, 20).iloc[-1]
    cci_state = "overbought" if cci20 > 100 else "oversold" if cci20 < -100 else "neutral"
    _s("Momentum", "CCI(20)", cci20, cci_state,
       f"CCI={cci20:.0f} ({'Overbought' if cci20 > 100 else 'Oversold' if cci20 < -100 else 'Neutral'})")

    roc10 = (last_c / c.iloc[-11] - 1) * 100 if len(c) > 10 else 0.0
    _s("Momentum", "ROC(10)", roc10,
       "bullish" if roc10 > 0 else "bearish", f"10d price change: {roc10:+.2f}%")

    # ── VOLATILITY ────────────────────────────────────────────────────────────
    boll_mid_v = c.rolling(20).mean().iloc[-1]
    boll_std_v = c.rolling(20).std().iloc[-1]
    pct_b = (last_c - (boll_mid_v - 2 * boll_std_v)) / (4 * boll_std_v + 1e-9)
    boll_state = "overbought" if pct_b > 1 else "oversold" if pct_b < 0 else "neutral"
    _s("Volatility", "Bollinger %B", pct_b * 100, boll_state,
       f"%B={pct_b*100:.0f}% ({'Above upper band' if pct_b > 1 else 'Below lower band' if pct_b < 0 else 'Within bands'})")

    atr14_v  = _atr(h, lo, c, 14).iloc[-1]
    atr_norm = atr14_v / last_c * 100
    hv21     = _log_return(c, 1).rolling(21).std().iloc[-1] * np.sqrt(252) * 100
    hv_med   = _log_return(c, 1).rolling(21).std().rolling(252).median().iloc[-1] * np.sqrt(252) * 100
    vol_state = "high_vol" if hv21 > hv_med * 1.3 else "low_vol" if hv21 < hv_med * 0.7 else "neutral"
    _s("Volatility", "Historical Vol 21d", hv21, vol_state,
       f"HV21={hv21:.1f}%  (median={hv_med:.1f}%)")

    kelt_pct = _keltner_pct(h, lo, c, 20, 10, 2.0).iloc[-1] * 100
    kelt_state = "overbought" if kelt_pct > 100 else "oversold" if kelt_pct < 0 else "neutral"
    _s("Volatility", "Keltner Channel %", kelt_pct, kelt_state,
       f"Position in channel: {kelt_pct:.0f}%")

    # ── VOLUME ────────────────────────────────────────────────────────────────
    if v is not None and v.notna().sum() > 50:
        v_z = ((v - v.rolling(21).mean()) / (v.rolling(21).std() + 1e-9)).iloc[-1]
        vol_sig = "high_vol" if v_z > 1.5 else "low_vol" if v_z < -1.5 else "neutral"
        _s("Volume", "Volume Z-score 21d", v_z, vol_sig,
           f"Z={v_z:.2f} ({'High volume' if v_z > 1.5 else 'Low volume' if v_z < -1.5 else 'Normal volume'})")

        cmf20_v = _cmf(h, lo, c, v, 20).iloc[-1]
        cmf_state = "bullish" if cmf20_v > 0.05 else "bearish" if cmf20_v < -0.05 else "neutral"
        _s("Volume", "CMF(20)", cmf20_v, cmf_state,
           f"CMF={cmf20_v:.3f} ({'Buying pressure' if cmf_state == 'bullish' else 'Selling pressure' if cmf_state == 'bearish' else 'Neutral'})")

        mfi14_v = _mfi(h, lo, c, v, 14).iloc[-1]
        mfi_state = "overbought" if mfi14_v > 80 else "oversold" if mfi14_v < 20 else "neutral"
        _s("Volume", "MFI(14)", mfi14_v, mfi_state,
           f"MFI={mfi14_v:.1f} ({'Overbought' if mfi14_v > 80 else 'Oversold' if mfi14_v < 20 else 'Neutral'})")

    # ── PIVOTS ────────────────────────────────────────────────────────────────
    res20 = h.rolling(20).max().shift(1).iloc[-1]
    sup20 = lo.rolling(20).min().shift(1).iloc[-1]
    dist_res = (last_c - res20) / last_c * 100
    dist_sup = (last_c - sup20) / last_c * 100
    pivot_state = "near_resistance" if dist_res > -1.0 else \
                  "near_support"    if dist_sup < 1.0  else "middle"
    _s("Pivots", "Resistance 20d", dist_res, pivot_state,
       f"Distance to resistance: {dist_res:+.1f}%")
    _s("Pivots", "Support 20d", dist_sup, pivot_state,
       f"Distance to support: {dist_sup:+.1f}%")

    ph = h.shift(1).iloc[-1]
    pl = lo.shift(1).iloc[-1]
    pc_prev = c.shift(1).iloc[-1]
    piv_point = (ph + pl + pc_prev) / 3
    dist_piv = (last_c - piv_point) / last_c * 100
    _s("Pivots", "Pivot Point", dist_piv,
       "bullish" if dist_piv > 0 else "bearish",
       f"Price {'above' if dist_piv > 0 else 'below'} pivot ({dist_piv:+.1f}%)")

    return states
