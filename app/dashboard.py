"""
IBEX35 Forecasting Dashboard

Tabs:
  1. Signal       — Current market signal (UP/DOWN/NEUTRAL + probability)
  2. Market       — Candlestick, SMAs, Bollinger Bands, support/resistance
  3. Indicators   — All 4 technical categories + pivot states with badges
  4. Explainability — SHAP feature importance + latest prediction drivers
  5. Results      — Walk-forward ML + economic metrics comparison
  6. Backtest     — Equity curves + drawdown

Run with:
    streamlit run app/dashboard.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Make src importable when running from project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st

from src.data.fetch import load_raw
from src.data.features import build_features, feature_cols, get_indicator_states
from src.data.multiasset import fetch_multiasset_features
from src.utils.config import get, root

st.set_page_config(
    page_title="IBEX35 Forecast",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("IBEX35 Forecast")
    st.caption("Walk-forward ML forecasting system")
    st.divider()
    if st.button("Refresh data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    st.divider()
    st.markdown("""
**How to use**
1. `python train.py` — train all models
2. `python predict.py --shap` — generate latest signal
3. Open this dashboard

**Workflow**
- Walk-forward validation (no data leakage)
- Transaction costs: 10 bps per trade
- Confidence threshold: 55% for a signal
""")
    st.divider()
    st.caption("Models: XGBoost · LightGBM · Logistic · Random Forest")
    st.caption("Data: Yahoo Finance (^IBEX)")


# ─────────────────────────────────────────────────────────────────────────────
# Cached data loaders
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def get_data():
    df_raw  = load_raw()
    df_feat = build_features(df_raw)
    ma      = fetch_multiasset_features(df_feat.index)
    if not ma.empty:
        df_feat = df_feat.join(ma, how="left")
    return df_raw, df_feat


@st.cache_data(ttl=60, show_spinner=False)
def load_latest_prediction() -> dict:
    path = root() / "results" / "latest_prediction.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


@st.cache_data(ttl=60, show_spinner=False)
def load_results() -> dict:
    results_dir = root() / "results"
    out = {}
    if results_dir.exists():
        for p in results_dir.glob("*_summary.json"):
            with open(p) as f:
                out[p.stem.replace("_summary", "")] = json.load(f)
    return out


@st.cache_data(ttl=60, show_spinner=False)
def load_predictions_df(target: str, model_name: str) -> pd.DataFrame | None:
    path = root() / "results" / f"{target}_{model_name}_predictions.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────────────────────────────────────

with st.spinner("Loading market data..."):
    df_raw, df_feat = get_data()

latest_pred      = load_latest_prediction()
results          = load_results()
feat_c           = feature_cols(df_feat)
indicator_states = get_indicator_states(df_raw)

# ─────────────────────────────────────────────────────────────────────────────
# Header — key market metrics
# ─────────────────────────────────────────────────────────────────────────────

last_c   = float(df_raw["Close"].iloc[-1])
prev_c   = float(df_raw["Close"].iloc[-2])
chg_pct  = (last_c / prev_c - 1) * 100
last_dt  = df_raw.index[-1].strftime("%d %b %Y")
hv21     = float(np.log(df_raw["Close"] / df_raw["Close"].shift(1)).rolling(21).std().iloc[-1] * np.sqrt(252) * 100)

st.markdown(f"## IBEX35 &nbsp;&nbsp; <span style='color:gray;font-size:18px'>{last_dt}</span>", unsafe_allow_html=True)
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Last Price",        f"{last_c:,.0f}")
c2.metric("1d Change",         f"{chg_pct:+.2f}%",   delta=f"{chg_pct:+.2f}%")
c3.metric("HV 21d (ann.)",     f"{hv21:.1f}%")
c4.metric("Trading days",      f"{len(df_raw):,}")
c5.metric("Features",          f"{len(feat_c)}")

# ─────────────────────────────────────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────────────────────────────────────

def _thr_badge(thr: dict) -> str:
    """Return a small inline HTML badge showing optimisation holdout Sharpe if available."""
    if thr.get("mode") == "optimized_percentile" and "holdout_sharpe" in thr:
        sh = thr["holdout_sharpe"]
        col = "#2e7d32" if sh > 0.5 else "#e65100" if sh > 0 else "#c62828"
        return (f'&nbsp;·&nbsp; <span style="color:{col}">holdout Sharpe {sh:+.2f}</span>')
    return ""


t_summary, t_signal, t_market, t_indicators, t_explain, t_results, t_backtest = st.tabs([
    "📝 Summary", "📊 Signal", "📈 Market", "🔧 Indicators", "🧠 Explainability", "📋 Results", "💰 Backtest",
])


# ═════════════════════════════════════════════════════════════════════════════
# TAB 0 — OPERATIVE SUMMARY
# ═════════════════════════════════════════════════════════════════════════════
with t_summary:
    st.subheader("Operative Summary")
    st.caption("Narrative explanation of the current prediction, sentiment state, and media coverage.")

    col_opts, col_run = st.columns([3, 1])
    with col_opts:
        include_news = st.checkbox("Include live news sentiment", value=True,
                                   help="Fetches current RSS headlines. Takes 5-15s on first run.")
    with col_run:
        run_summary = st.button("Generate summary", use_container_width=True, type="primary")

    # Load cached summary or generate on demand
    @st.cache_data(ttl=900, show_spinner=False)  # 15-min cache
    def _cached_summary(include_news: bool) -> dict:
        from app.summary import generate_summary
        return generate_summary(include_live_news=include_news)

    summary_data: dict = {}
    if run_summary:
        st.cache_data.clear()
        with st.spinner("Generating operative summary (fetching news + scoring)..."):
            from app.summary import generate_summary
            summary_data = generate_summary(include_live_news=include_news)
    else:
        # Try to load saved summary
        summary_path = root() / "results" / "latest_summary.json"
        if summary_path.exists():
            with open(summary_path) as _f:
                summary_data = json.load(_f)

    if not summary_data:
        st.info("Click **Generate summary** to produce the operative summary, or run `python predict.py --shap`.")
    else:
        # ── Narrative ──────────────────────────────────────────────────────
        narrative = summary_data.get("narrative_md", "")
        if narrative:
            st.markdown(narrative, unsafe_allow_html=True)

        # ── Sentiment detail table ─────────────────────────────────────────
        sent = summary_data.get("sentiment", {})
        if sent:
            st.divider()
            st.markdown("#### Sentiment layers")
            s_col1, s_col2, s_col3, s_col4 = st.columns(4)
            s_col1.metric("Composite",       f"{sent.get('composite', 0):+.3f}")
            s_col2.metric("Direct IBEX",     f"{sent.get('direct_ibex', 0):+.3f}")
            s_col3.metric("Constituent",     f"{sent.get('constituent_rollup', 0):+.3f}")
            s_col4.metric("Macro / ECB",     f"{sent.get('macro', 0):+.3f}")

        # ── Media table ────────────────────────────────────────────────────
        media = summary_data.get("media", [])
        if media:
            st.divider()
            st.markdown("#### Recent media coverage")
            SENT_COLOR = {"pos": "#d4edda", "neu": "#fff3cd", "neg": "#f8d7da"}
            LAYER_EMOJI = {"index": "📊", "constituent": "🏢", "macro": "🌍"}

            df_media = pd.DataFrame(media)[
                ["published", "source", "headline", "target_type", "target_name",
                 "sentiment_score", "sentiment_label"]
            ].rename(columns={
                "published": "Published", "source": "Source",
                "headline": "Headline", "target_type": "Layer",
                "target_name": "Target", "sentiment_score": "Score",
                "sentiment_label": "Tone",
            })
            df_media["Layer"] = df_media["Layer"].map(LAYER_EMOJI).fillna("—") + " " + df_media["Layer"]

            st.dataframe(
                df_media,
                width="stretch",
                height=min(400, 35 + 35 * len(df_media)),
                column_config={
                    "Score": st.column_config.NumberColumn(format="%.3f"),
                    "Published": st.column_config.TextColumn(width="medium"),
                    "Headline": st.column_config.TextColumn(width="large"),
                },
                hide_index=True,
            )

        st.caption(f"Summary generated: {summary_data.get('generated_at', '—')}")


# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — SIGNAL
# ═════════════════════════════════════════════════════════════════════════════
with t_signal:
    st.subheader("Market Direction Signal")

    if not latest_pred:
        st.warning(
            "No prediction found. Run the pipeline first:\n\n"
            "```bash\npython train.py\npython predict.py --shap\n```"
        )
    else:
        col_1d, col_5d = st.columns(2)

        for col, key, horizon_label in [
            (col_1d, "target_dir_1d", "1-day horizon"),
            (col_5d, "target_dir_5d", "5-day horizon"),
        ]:
            with col:
                pred      = latest_pred.get(key, {})
                if "error" in pred:
                    st.error(f"Error: {pred['error']}")
                    continue

                signal    = pred.get("signal", "N/A")
                prob_up   = pred.get("prob_up",   0.5)
                prob_down = pred.get("prob_down",  0.5)
                conf      = pred.get("confidence", "?")
                model_n   = pred.get("model", "?")
                gen_at    = pred.get("generated_at", "?")

                color = {"UP": "#2e7d32", "DOWN": "#c62828", "NEUTRAL": "#546e7a"}.get(signal, "#546e7a")
                bg    = {"UP": "#e8f5e9",  "DOWN": "#ffebee",  "NEUTRAL": "#f5f7f8"}.get(signal, "#f5f7f8")
                arrow = {"UP": "↑",        "DOWN": "↓",        "NEUTRAL": "→"}.get(signal, "?")

                st.markdown(f"""
                <div style="text-align:center; padding:24px; border-radius:12px;
                            border: 2px solid {color}; background:{bg}; margin-bottom:8px">
                    <div style="font-size:13px; color:#666; margin-bottom:4px">{horizon_label}</div>
                    <div style="font-size:42px; font-weight:800; color:{color}; line-height:1">{arrow} {signal}</div>
                    <div style="font-size:16px; margin:8px 0">
                        P(up) = <b>{prob_up:.1%}</b> &nbsp;|&nbsp; P(down) = <b>{prob_down:.1%}</b>
                    </div>
                    <div style="font-size:12px; color:#888">
                        Confidence: <b>{conf}</b> &nbsp;·&nbsp; Model: {model_n} &nbsp;·&nbsp; {gen_at}
                        {_thr_badge(pred.get("signal_thresholds", {}))}
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Probability bar
                thr      = pred.get("signal_thresholds", {})
                up_thr   = thr.get("up_threshold",   0.55)
                down_thr = thr.get("down_threshold",  0.45)
                thr_mode = thr.get("mode", "fixed")
                sig_pct  = thr.get("signal_pct") or thr.get("optimal_pct", 0.75)
                if thr_mode == "optimized_percentile":
                    thr_label = f"Sharpe-optimised ({(1-sig_pct):.0%} active)"
                elif thr_mode == "percentile":
                    thr_label = f"top/bottom {(1-sig_pct):.0%} of OOS distribution"
                else:
                    thr_label = "fixed threshold"

                fig_bar = go.Figure(go.Bar(
                    x=["P(up)", "P(down)"],
                    y=[prob_up * 100, prob_down * 100],
                    marker_color=[
                        "#4caf50" if prob_up  >= up_thr   else "#ff9800",
                        "#f44336" if prob_down >= (1 - down_thr) else "#ff9800",
                    ],
                    text=[f"{prob_up:.1%}", f"{prob_down:.1%}"],
                    textposition="outside",
                ))
                fig_bar.add_hline(
                    y=up_thr * 100, line_dash="dot", line_color="#4caf50",
                    annotation_text=f"UP ≥ {up_thr:.1%} ({thr_label})",
                    annotation_position="top right",
                )
                fig_bar.add_hline(
                    y=(1 - down_thr) * 100, line_dash="dot", line_color="#f44336",
                    annotation_text=f"DOWN ≥ {(1-down_thr):.1%}",
                    annotation_position="bottom right",
                )
                fig_bar.update_layout(
                    height=260, margin=dict(t=10, b=10, l=0, r=0),
                    yaxis_range=[0, 100], showlegend=False,
                    yaxis_title="Probability (%)",
                )
                st.plotly_chart(fig_bar, width="stretch", key=f"sig_bar_{key}")

        st.divider()

        # SHAP top drivers
        for key, horizon_label in [("target_dir_1d", "1-day"), ("target_dir_5d", "5-day")]:
            pred    = latest_pred.get(key, {})
            drivers = pred.get("top_drivers", [])
            if not drivers:
                continue
            st.subheader(f"Key SHAP drivers — {horizon_label}")
            st.caption("Green bars push the prediction toward UP; red bars toward DOWN.")
            df_d = pd.DataFrame(drivers)
            fig_d = px.bar(
                df_d.head(10), x="shap_value", y="feature", orientation="h",
                color="shap_value",
                color_continuous_scale=["#f44336", "#ffffff", "#4caf50"],
                color_continuous_midpoint=0,
                labels={"shap_value": "SHAP impact", "feature": ""},
            )
            fig_d.update_layout(
                height=320, margin=dict(t=10, b=0, l=0, r=0),
                showlegend=False, coloraxis_showscale=False,
            )
            st.plotly_chart(fig_d, width="stretch", key=f"sig_shap_{key}")


# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — MARKET OVERVIEW
# ═════════════════════════════════════════════════════════════════════════════
with t_market:
    ctrl_col, _ = st.columns([3, 1])
    with ctrl_col:
        n_days = st.slider("Days to display", 60, 2000, 500, key="mkt_days")

    df_plot = df_raw.tail(n_days).copy()

    ov_cols = st.columns(5)
    show_sma20  = ov_cols[0].checkbox("SMA 20",           value=True)
    show_sma50  = ov_cols[1].checkbox("SMA 50",           value=True)
    show_sma200 = ov_cols[2].checkbox("SMA 200",          value=True)
    show_boll   = ov_cols[3].checkbox("Bollinger Bands",  value=False)
    show_sr     = ov_cols[4].checkbox("Support/Resistance (20d)", value=True)

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.75, 0.25], vertical_spacing=0.04,
        subplot_titles=["IBEX35 Price", "Volume"],
    )

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df_plot.index, open=df_plot["Open"], high=df_plot["High"],
        low=df_plot["Low"], close=df_plot["Close"],
        name="IBEX35", increasing_line_color="#26a69a",
        decreasing_line_color="#ef5350",
    ), row=1, col=1)

    if show_sma20:
        fig.add_trace(go.Scatter(x=df_plot.index,
            y=df_plot["Close"].rolling(20).mean(),
            name="SMA20", line=dict(color="#2196f3", width=1.5)), row=1, col=1)
    if show_sma50:
        fig.add_trace(go.Scatter(x=df_plot.index,
            y=df_plot["Close"].rolling(50).mean(),
            name="SMA50", line=dict(color="#ff9800", width=1.5)), row=1, col=1)
    if show_sma200:
        fig.add_trace(go.Scatter(x=df_plot.index,
            y=df_plot["Close"].rolling(200).mean(),
            name="SMA200", line=dict(color="#9c27b0", width=1.5, dash="dot")), row=1, col=1)
    if show_boll:
        bm = df_plot["Close"].rolling(20).mean()
        bs = df_plot["Close"].rolling(20).std()
        fig.add_trace(go.Scatter(x=df_plot.index, y=bm + 2*bs,
            name="BB Upper", line=dict(color="#78909c", width=1, dash="dot"),
            fill=None), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_plot.index, y=bm - 2*bs,
            name="BB Lower", line=dict(color="#78909c", width=1, dash="dot"),
            fill="tonexty", fillcolor="rgba(120,144,156,0.10)"), row=1, col=1)
    if show_sr:
        res = df_plot["High"].rolling(20).max().shift(1)
        sup = df_plot["Low"].rolling(20).min().shift(1)
        fig.add_trace(go.Scatter(x=df_plot.index, y=res,
            name="Resistance 20d", line=dict(color="#f44336", width=1, dash="dash")), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_plot.index, y=sup,
            name="Support 20d", line=dict(color="#4caf50", width=1, dash="dash")), row=1, col=1)

    vol_colors = ["#26a69a" if df_plot["Close"].iloc[i] >= df_plot["Open"].iloc[i]
                  else "#ef5350" for i in range(len(df_plot))]
    fig.add_trace(go.Bar(x=df_plot.index, y=df_plot["Volume"],
        name="Volume", marker_color=vol_colors, opacity=0.7), row=2, col=1)

    fig.update_layout(
        height=640, xaxis_rangeslider_visible=False,
        margin=dict(t=30, b=20, l=0, r=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    fig.update_xaxes(showgrid=False)
    st.plotly_chart(fig, width="stretch", key="market_candle")


# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 — TECHNICAL INDICATORS
# ═════════════════════════════════════════════════════════════════════════════
with t_indicators:
    st.subheader("Current Technical Indicator States")

    STATE_COLOR = {
        "bullish":         "#2e7d32",
        "bearish":         "#c62828",
        "neutral":         "#546e7a",
        "overbought":      "#e65100",
        "oversold":        "#1565c0",
        "high_vol":        "#ad1457",
        "low_vol":         "#00838f",
        "near_resistance": "#bf360c",
        "near_support":    "#558b2f",
        "middle":          "#546e7a",
    }
    STATE_LABEL = {
        "bullish":         "Bullish",
        "bearish":         "Bearish",
        "neutral":         "Neutral",
        "overbought":      "Overbought",
        "oversold":        "Oversold",
        "high_vol":        "High Volatility",
        "low_vol":         "Low Volatility",
        "near_resistance": "Near Resistance",
        "near_support":    "Near Support",
        "middle":          "In Range",
    }

    if not indicator_states:
        st.warning("Insufficient data to compute indicators (need >210 days).")
    else:
        categories = ["Trend", "Momentum", "Volatility", "Volume", "Pivots"]
        for cat in categories:
            cat_states = [s for s in indicator_states if s["category"] == cat]
            if not cat_states:
                continue
            st.markdown(f"#### {cat}")
            n_cols = min(4, len(cat_states))
            cols   = st.columns(n_cols)
            for i, state in enumerate(cat_states):
                with cols[i % n_cols]:
                    color = STATE_COLOR.get(state["state"], "#546e7a")
                    label = STATE_LABEL.get(state["state"], state["state"])
                    val   = f"{state['value']:.2f}" if state["value"] is not None else "N/A"
                    st.markdown(f"""
                    <div style="border-left: 4px solid {color}; padding:10px 14px; margin-bottom:10px;
                                background:#fafafa; border-radius:4px; box-shadow:0 1px 3px rgba(0,0,0,0.08)">
                        <div style="font-size:11px; color:#888; text-transform:uppercase; letter-spacing:.5px">{state['name']}</div>
                        <div style="font-size:18px; font-weight:700; color:{color}; margin:2px 0">{label}</div>
                        <div style="font-size:12px; color:#555">{state['label']}</div>
                        <div style="font-size:11px; color:#aaa; margin-top:2px">value: {val}</div>
                    </div>
                    """, unsafe_allow_html=True)
            st.divider()

    # ── Interactive feature chart ─────────────────────────────────────────────
    st.subheader("Feature Explorer")
    default_feats = ["rsi_14", "macd_hist", "stoch_k", "boll_pct_b", "adx14", "cmf20"]
    default_feats = [f for f in default_feats if f in feat_c][:3]
    selected = st.multiselect("Features to plot", feat_c, default=default_feats,
                               help="Select one or more features to compare over time")
    n_hist = st.slider("History (days)", 60, 1000, 252, key="ind_days")
    if selected:
        fig_ind = go.Figure()
        df_ind  = df_feat[selected].tail(n_hist)
        for col_name in selected:
            fig_ind.add_trace(go.Scatter(
                x=df_ind.index, y=df_ind[col_name], name=col_name, mode="lines",
            ))
        fig_ind.update_layout(
            height=360, margin=dict(t=10, b=0, l=0, r=0),
            legend=dict(orientation="h"),
        )
        st.plotly_chart(fig_ind, width="stretch", key="ind_feature_explorer")


# ═════════════════════════════════════════════════════════════════════════════
# TAB 4 — EXPLAINABILITY
# ═════════════════════════════════════════════════════════════════════════════
with t_explain:
    st.subheader("SHAP Explainability")
    st.caption(
        "SHAP (SHapley Additive exPlanations) measures each feature's contribution "
        "to a specific prediction. Positive values push toward UP; negative toward DOWN."
    )

    col_sel1, col_sel2 = st.columns(2)
    target_exp = col_sel1.selectbox(
        "Target", ["target_dir_1d", "target_dir_5d"], key="exp_target",
    )
    model_exp  = col_sel2.selectbox(
        "Model", ["xgboost_cal", "xgboost", "lgbm", "random_forest", "logistic"], key="exp_model",
    )

    if st.button("Compute SHAP (~30s)", type="primary"):
        with st.spinner("Computing SHAP values..."):
            try:
                from src.models.ensemble import load_model
                from src.explainability.shap_analysis import compute_shap, shap_summary, shap_for_latest

                model  = load_model(model_exp, target_exp)
                df_t   = df_feat.dropna(subset=[target_exp])
                X      = df_t[feat_c].values
                shap_vals, f_names = compute_shap(model, X, feat_c, max_samples=500)
                df_imp = shap_summary(shap_vals, f_names, top_n=25)

                st.success(f"SHAP computed on {len(shap_vals):,} samples · {len(f_names)} features")

                col_imp, col_water = st.columns(2)
                with col_imp:
                    st.markdown("**Global feature importance (mean |SHAP|)**")
                    fig_imp = px.bar(
                        df_imp, x="mean_shap", y="feature", orientation="h",
                        color="mean_shap", color_continuous_scale="Blues",
                        labels={"mean_shap": "Mean |SHAP|", "feature": ""},
                    )
                    fig_imp.update_layout(height=600, margin=dict(t=10, b=0, l=0, r=0),
                                          coloraxis_showscale=False)
                    st.plotly_chart(fig_imp, width="stretch", key="shap_global_imp")

                with col_water:
                    st.markdown("**Latest prediction drivers (waterfall)**")
                    latest_x = df_t[feat_c].dropna().iloc[-1].values
                    df_shap_latest = shap_for_latest(model, latest_x, feat_c)
                    df_top = df_shap_latest.head(15)

                    fig_wf = go.Figure(go.Bar(
                        x=df_top["shap_value"],
                        y=df_top["feature"],
                        orientation="h",
                        marker_color=["#2e7d32" if v > 0 else "#c62828"
                                      for v in df_top["shap_value"]],
                        text=df_top["shap_value"].round(4),
                        textposition="outside",
                    ))
                    fig_wf.update_layout(
                        height=500, margin=dict(t=10, b=0, l=0, r=0),
                        xaxis_title="SHAP impact (green=bullish, red=bearish)",
                        yaxis_title="",
                    )
                    st.plotly_chart(fig_wf, width="stretch", key="shap_waterfall")

                with st.expander("Full feature table"):
                    st.dataframe(df_shap_latest.round(4), width="stretch")

            except FileNotFoundError as e:
                st.error(f"Model not found: {e}\n\nRun `python train.py` first.")
            except ImportError as e:
                st.error(f"Missing dependency: {e}\n\nRun: `pip install shap`")
            except Exception as e:
                st.error(f"Error: {e}")


# ═════════════════════════════════════════════════════════════════════════════
# TAB 5 — MODEL RESULTS
# ═════════════════════════════════════════════════════════════════════════════
with t_results:
    if not results:
        st.info("No results found. Run `python train.py` to train models.")
    else:
        target_r = st.selectbox("Target", list(results.keys()), key="res_target")
        r        = results[target_r]

        ml_keys = ["accuracy", "balanced_accuracy", "f1", "f1_macro", "roc_auc", "pr_auc",
                   "mae", "rmse", "r2", "ic"]

        col_ml, col_eco = st.columns(2)

        with col_ml:
            st.markdown("**ML Metrics (walk-forward mean)**")
            st.caption("Averages across all out-of-sample validation folds.")
            rows = []
            for model, metrics in r.items():
                row = {"model": model}
                row.update({k: metrics[k] for k in ml_keys if k in metrics})
                rows.append(row)
            df_ml = pd.DataFrame(rows).set_index("model")
            if not df_ml.empty:
                st.dataframe(
                    df_ml.style.format("{:.4f}")
                    .highlight_max(axis=0, color="#c8e6c9")
                    .highlight_min(axis=0, color="#ffcdd2"),
                    width="stretch",
                )

        with col_eco:
            bt_keys = ["bt_ann_return", "bt_ann_volatility", "bt_sharpe",
                       "bt_max_drawdown", "bt_calmar", "bt_hit_ratio",
                       "bt_bm_ann_return", "bt_excess_return"]
            st.markdown("**Economic Metrics (walk-forward mean)**")
            st.caption("Strategy performance after transaction costs vs buy-and-hold.")
            rows_bt = []
            for model, metrics in r.items():
                row = {"model": model}
                row.update({k.replace("bt_", ""): metrics[k] for k in bt_keys if k in metrics})
                rows_bt.append(row)
            df_bt = pd.DataFrame(rows_bt).set_index("model")
            if not df_bt.empty:
                highlight_pos = [c for c in df_bt.columns if c in
                                 ["ann_return", "sharpe", "calmar", "hit_ratio", "excess_return"]]
                highlight_neg = [c for c in df_bt.columns if c in ["max_drawdown", "ann_volatility"]]
                sty = df_bt.style.format("{:.4f}")
                if highlight_pos:
                    sty = sty.highlight_max(axis=0, subset=highlight_pos, color="#c8e6c9")
                if highlight_neg:
                    sty = sty.highlight_min(axis=0, subset=highlight_neg, color="#c8e6c9")
                st.dataframe(sty, width="stretch")

        # Walk-forward predictions timeline
        st.markdown("**Walk-forward prediction timeline**")
        model_sel = st.selectbox("Model", list(r.keys()), key="pred_model")
        pred_df   = load_predictions_df(target_r, model_sel)
        if pred_df is not None and "y_prob" in pred_df.columns:
            fig_pred = go.Figure()
            fig_pred.add_trace(go.Scatter(
                x=pred_df.index, y=pred_df["y_prob"],
                name="P(up)", line=dict(color="#2196f3"),
            ))
            fig_pred.add_hline(y=0.5, line_dash="dash", line_color="gray",
                               annotation_text="50%")
            # Load model-specific UP threshold
            _thr_path = root() / "results" / "signal_thresholds.json"
            _thr_key  = f"{target_r}_{model_sel}"
            _up_thr   = get("backtest.min_confidence", 0.55)
            if _thr_path.exists():
                import json as _json
                _thr_all = _json.load(open(_thr_path))
                _up_thr  = _thr_all.get(_thr_key, {}).get("up_threshold", _up_thr)
            fig_pred.add_hline(y=_up_thr, line_dash="dot", line_color="green",
                               annotation_text=f"UP threshold ({_up_thr:.1%})")
            correct   = pred_df[pred_df["y_true"] == pred_df["y_pred"]]
            incorrect = pred_df[pred_df["y_true"] != pred_df["y_pred"]]
            fig_pred.add_trace(go.Scatter(
                x=correct.index, y=correct["y_prob"],
                mode="markers", name="Correct",
                marker=dict(color="#4caf50", size=5, symbol="circle"),
            ))
            fig_pred.add_trace(go.Scatter(
                x=incorrect.index, y=incorrect["y_prob"],
                mode="markers", name="Incorrect",
                marker=dict(color="#f44336", size=5, symbol="x"),
            ))
            fig_pred.update_layout(
                height=380, margin=dict(t=10, b=0, l=0, r=0),
                yaxis_range=[0, 1], yaxis_title="P(up)",
                legend=dict(orientation="h"),
            )
            st.plotly_chart(fig_pred, width="stretch", key="res_pred_timeline")


# ═════════════════════════════════════════════════════════════════════════════
# TAB 6 — BACKTEST
# ═════════════════════════════════════════════════════════════════════════════
with t_backtest:
    if not results:
        st.info("No results found. Run `python train.py` first.")
    else:
        target_bt = st.selectbox("Target", list(results.keys()), key="bt_target")
        r_bt      = results[target_bt]

        bt_col1, bt_col2 = st.columns([1, 2])
        with bt_col1:
            # Default to champion model's percentile UP-threshold when available
            _bt_thr_path = root() / "results" / "signal_thresholds.json"
            _bt_default  = get("backtest.min_confidence", 0.55)
            if _bt_thr_path.exists():
                import json as _json2
                _bt_all     = _json2.load(open(_bt_thr_path))
                _bt_key     = f"{target_bt}_{list(r_bt.keys())[0]}"
                _bt_default = float(_bt_all.get(_bt_key, {}).get("up_threshold", _bt_default))
            min_conf_bt = st.slider("Confidence threshold", 0.50, 0.70,
                                    float(round(_bt_default, 2)), step=0.01)
            tc_bps_bt   = st.slider("Transaction cost (bps)", 0, 30, 10)
            st.caption("Strategy goes long when P(up) ≥ threshold; flat otherwise.")

        with bt_col2:
            selected_models = st.multiselect(
                "Models to compare", list(r_bt.keys()),
                default=list(r_bt.keys())[:3],
            )

        fig_eq = make_subplots(
            rows=2, cols=1, shared_xaxes=True,
            row_heights=[0.65, 0.35], vertical_spacing=0.04,
            subplot_titles=["Equity curve (walk-forward, after costs)", "Drawdown (%)"],
        )
        has_bm    = False
        bm_index  = None
        bt_stats  = []

        for model_name in selected_models:
            pred_df = load_predictions_df(target_bt, model_name)
            if pred_df is None or "y_prob" not in pred_df.columns:
                continue
            ret_col = "target_ret_5d" if "5d" in target_bt and "target_ret_5d" in df_feat.columns \
                      else "target_ret_1d"
            ret_series = df_feat[ret_col].reindex(pred_df.index).fillna(0) \
                         if ret_col in df_feat.columns \
                         else pd.Series(0.0, index=pred_df.index)

            positions  = np.where(pred_df["y_prob"] >= min_conf_bt, 1.0, 0.0)
            trades     = np.abs(np.diff(positions, prepend=0.0))
            # Log returns → compound with exp(cumsum); subtract simple transaction costs
            log_ret    = ret_series.values
            strat_ret  = positions * log_ret - trades * tc_bps_bt / 10_000
            equity     = np.exp(np.cumsum(strat_ret))
            drawdown   = (equity / np.maximum.accumulate(equity) - 1) * 100

            ann_ret   = float(np.exp(strat_ret.mean() * 252) - 1)
            ann_vol   = float(strat_ret.std() * np.sqrt(252))
            sharpe    = ann_ret / ann_vol if ann_vol > 0 else 0.0
            max_dd    = float(drawdown.min())
            bt_stats.append({"Model": model_name,
                              "Ann. Return": f"{ann_ret:.1%}",
                              "Ann. Vol": f"{ann_vol:.1%}",
                              "Sharpe": f"{sharpe:.2f}",
                              "Max DD": f"{max_dd:.1f}%"})

            fig_eq.add_trace(go.Scatter(
                x=pred_df.index, y=equity, name=model_name, mode="lines",
            ), row=1, col=1)
            fig_eq.add_trace(go.Scatter(
                x=pred_df.index, y=drawdown, name=model_name,
                mode="lines", showlegend=False,
            ), row=2, col=1)

            if not has_bm:
                bm_log = ret_series.values
                bm_equity  = np.exp(np.cumsum(bm_log))
                bm_drawdown = (bm_equity / np.maximum.accumulate(bm_equity) - 1) * 100
                bm_ann_ret  = float(np.exp(bm_log.mean() * 252) - 1)
                bm_ann_vol  = float(bm_log.std() * np.sqrt(252))
                bm_sharpe   = bm_ann_ret / bm_ann_vol if bm_ann_vol > 0 else 0.0
                bm_maxdd    = float(bm_drawdown.min())
                bt_stats.append({"Model": "Buy & Hold",
                                  "Ann. Return": f"{bm_ann_ret:.1%}",
                                  "Ann. Vol": f"{bm_ann_vol:.1%}",
                                  "Sharpe": f"{bm_sharpe:.2f}",
                                  "Max DD": f"{bm_maxdd:.1f}%"})
                fig_eq.add_trace(go.Scatter(
                    x=pred_df.index, y=bm_equity, name="Buy & Hold", mode="lines",
                    line=dict(color="gray", dash="dot"),
                ), row=1, col=1)
                fig_eq.add_trace(go.Scatter(
                    x=pred_df.index, y=bm_drawdown, name="Buy & Hold",
                    mode="lines", showlegend=False,
                    line=dict(color="gray", dash="dot"),
                ), row=2, col=1)
                has_bm   = True
                bm_index = pred_df.index

        if fig_eq.data:
            fig_eq.update_layout(
                height=520, margin=dict(t=30, b=0, l=0, r=0),
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
            )
            fig_eq.update_yaxes(title_text="Portfolio value (base=1)", row=1, col=1)
            fig_eq.update_yaxes(title_text="Drawdown (%)", row=2, col=1)
            fig_eq.add_hline(y=0, line_dash="dot", line_color="red",
                             line_width=1, row=2, col=1)
            st.plotly_chart(fig_eq, width="stretch", key="bt_equity_curve")

            if bt_stats:
                st.markdown("**Summary statistics (out-of-sample)**")
                st.dataframe(pd.DataFrame(bt_stats).set_index("Model"),
                             width="stretch")
        else:
            st.info("Run `python train.py` to generate predictions.")
