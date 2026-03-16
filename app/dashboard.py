"""
Streamlit dashboard for IBEX35 forecasting.

Run with:  streamlit run app/dashboard.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from src.data.fetch import load_raw
from src.data.features import build_features, feature_cols, target_cols
from src.utils.config import root

st.set_page_config(
    page_title="IBEX35 Forecast",
    page_icon="📈",
    layout="wide",
)

# ── helpers ───────────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600)
def get_data():
    df_raw = load_raw()
    df_feat = build_features(df_raw)
    return df_raw, df_feat


def load_results() -> dict:
    results_dir = root() / "results"
    out = {}
    for p in results_dir.glob("*.json"):
        with open(p) as f:
            out[p.stem] = json.load(f)
    return out


# ── layout ────────────────────────────────────────────────────────────────────

st.title("IBEX35 Forecasting System")

with st.spinner("Loading data..."):
    df_raw, df_feat = get_data()

results = load_results()

tab_overview, tab_features, tab_results, tab_backtest = st.tabs(
    ["Market Overview", "Features", "Model Results", "Backtest"]
)

# ── Tab 1: Overview ───────────────────────────────────────────────────────────
with tab_overview:
    col1, col2, col3 = st.columns(3)
    last  = df_raw["Close"].iloc[-1]
    prev  = df_raw["Close"].iloc[-2]
    ret1d = (last / prev - 1) * 100
    ret_ytd = (last / df_raw.loc[df_raw.index.year == df_raw.index[-1].year, "Close"].iloc[0] - 1) * 100

    col1.metric("IBEX35", f"{last:,.0f}", f"{ret1d:+.2f}% 1d")
    col2.metric("YTD Return", f"{ret_ytd:+.1f}%")
    col3.metric("Data rows", f"{len(df_raw):,}")

    # Price chart
    n_days = st.slider("Days to show", 60, 1500, 500)
    df_plot = df_raw.tail(n_days)
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df_plot.index, open=df_plot["Open"],
        high=df_plot["High"], low=df_plot["Low"], close=df_plot["Close"],
        name="IBEX35",
    ))
    fig.update_layout(height=450, xaxis_rangeslider_visible=False,
                      title="IBEX35 Price (Candlestick)")
    st.plotly_chart(fig, use_container_width=True)

# ── Tab 2: Features ───────────────────────────────────────────────────────────
with tab_features:
    feat_c = feature_cols(df_feat)
    selected = st.multiselect("Features to plot", feat_c,
                              default=["rsi_14", "macd_hist", "vol_z21"][:3])
    if selected:
        fig = px.line(df_feat.tail(500), y=selected, title="Feature time series")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Feature statistics (last 252 trading days)")
    st.dataframe(df_feat[feat_c].tail(252).describe().round(4))

# ── Tab 3: Model Results ──────────────────────────────────────────────────────
with tab_results:
    if not results:
        st.info("No results yet. Run `python train.py` first.")
    else:
        target_choice = st.selectbox("Target", list(results.keys()))
        r = results[target_choice]
        rows = []
        for model, metrics in r.items():
            row = {"model": model}
            row.update(metrics)
            rows.append(row)
        df_res = pd.DataFrame(rows).set_index("model")
        st.dataframe(df_res.style.highlight_max(axis=0, color="#d4edda")
                                  .highlight_min(axis=0, color="#f8d7da")
                                  .format("{:.4f}"))

# ── Tab 4: Backtest ───────────────────────────────────────────────────────────
with tab_backtest:
    st.subheader("Walk-forward backtest economics")
    if not results:
        st.info("Run `python train.py` first.")
    else:
        target_choice = st.selectbox("Target ", list(results.keys()), key="bt_target")
        r = results[target_choice]
        bt_keys = [k for k in list(r.values())[0] if k.startswith("bt_")]
        if bt_keys:
            bt_rows = []
            for model, metrics in r.items():
                row = {"model": model}
                row.update({k: metrics[k] for k in bt_keys if k in metrics})
                bt_rows.append(row)
            st.dataframe(pd.DataFrame(bt_rows).set_index("model").round(4))
        else:
            st.info("Backtest metrics only available for classification targets.")
