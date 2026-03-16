"""
IBEX35 Forecasting Dashboard

Tabs:
  1. Señal          — Current market signal (UP/DOWN/NEUTRAL + probability)
  2. Mercado        — Candlestick, SMAs, volume, support/resistance
  3. Indicadores    — All 4 technical categories + pivot states with badges
  4. Explicabilidad — SHAP feature importance + latest prediction drivers
  5. Resultados     — Walk-forward ML + economic metrics comparison
  6. Backtest       — Equity curves + drawdown

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
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

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
    pred_dir = root() / "results"
    path     = pred_dir / f"{target}_{model_name}_predictions.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────────────────────────────────────

with st.spinner("Cargando datos..."):
    df_raw, df_feat = get_data()

latest_pred = load_latest_prediction()
results     = load_results()
feat_c      = feature_cols(df_feat)
indicator_states = get_indicator_states(df_raw)

# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────

last_c  = float(df_raw["Close"].iloc[-1])
prev_c  = float(df_raw["Close"].iloc[-2])
chg_pct = (last_c / prev_c - 1) * 100
last_dt = df_raw.index[-1].strftime("%d %b %Y")

st.markdown(f"## IBEX35 — {last_dt}")
col_px, col_chg, col_rows, col_feats = st.columns(4)
col_px.metric("Último precio", f"{last_c:,.0f}", f"{chg_pct:+.2f}%")
col_chg.metric("Variación 1d", f"{chg_pct:+.2f}%",
               delta_color="normal" if chg_pct >= 0 else "inverse")
col_rows.metric("Días de datos", f"{len(df_raw):,}")
col_feats.metric("Features", f"{len(feat_c)}")

# ─────────────────────────────────────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────────────────────────────────────

t_signal, t_market, t_indicators, t_explain, t_results, t_backtest = st.tabs([
    "Señal de Mercado", "Visión de Mercado", "Indicadores Técnicos",
    "Explicabilidad", "Resultados Modelos", "Backtest",
])


# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — MARKET SIGNAL
# ═════════════════════════════════════════════════════════════════════════════
with t_signal:
    st.subheader("¿Va a subir el IBEX35?")

    if not latest_pred:
        st.warning(
            "No hay predicción disponible. Ejecuta primero:\n\n"
            "```bash\npython train.py\npython predict.py --shap\n```"
        )
    else:
        col_1d, col_5d = st.columns(2)

        for col, key, horizon in [(col_1d, "target_dir_1d", "1 día"),
                                   (col_5d, "target_dir_5d", "5 días")]:
            with col:
                pred = latest_pred.get(key, {})
                if "error" in pred:
                    st.error(f"Error: {pred['error']}")
                    continue

                signal    = pred.get("signal", "N/A")
                prob_up   = pred.get("prob_up",   0.5)
                prob_down = pred.get("prob_down",  0.5)
                conf      = pred.get("confidence", "?")
                model_n   = pred.get("model", "?")

                # Color coding
                color = {"UP": "green", "DOWN": "red", "NEUTRAL": "gray"}.get(signal, "gray")
                icon  = {"UP": "↑",      "DOWN": "↓",  "NEUTRAL": "→"}.get(signal, "?")

                st.markdown(f"""
                <div style="text-align:center; padding:20px; border-radius:12px;
                            border: 2px solid {color}; background: {'#e8f5e9' if signal=='UP' else '#ffebee' if signal=='DOWN' else '#f5f5f5'}">
                    <h2 style="color:{color}; margin:0">{icon} {signal}</h2>
                    <h4 style="margin:4px 0">Horizonte: {horizon}</h4>
                    <p style="margin:4px 0">P(↑) = <b>{prob_up:.1%}</b> &nbsp;|&nbsp; P(↓) = <b>{prob_down:.1%}</b></p>
                    <p style="margin:4px 0; color:gray">Confianza: {conf} &nbsp;|&nbsp; Modelo: {model_n}</p>
                </div>
                """, unsafe_allow_html=True)

                # Probability bar
                fig_bar = go.Figure(go.Bar(
                    x=["P(↑ Sube)", "P(↓ Baja)"],
                    y=[prob_up * 100, prob_down * 100],
                    marker_color=[
                        "#4caf50" if prob_up >= 0.55 else "#ff9800",
                        "#f44336" if prob_down >= 0.55 else "#ff9800",
                    ],
                    text=[f"{prob_up:.1%}", f"{prob_down:.1%}"],
                    textposition="outside",
                ))
                fig_bar.add_hline(y=55, line_dash="dot", line_color="gray",
                                  annotation_text="umbral confianza 55%")
                fig_bar.update_layout(
                    height=260, margin=dict(t=20, b=20, l=0, r=0),
                    yaxis_range=[0, 100], showlegend=False,
                )
                st.plotly_chart(fig_bar, use_container_width=True)

        # SHAP top drivers
        for key, horizon in [("target_dir_1d", "1 día"), ("target_dir_5d", "5 días")]:
            pred = latest_pred.get(key, {})
            drivers = pred.get("top_drivers", [])
            if drivers:
                st.subheader(f"Factores clave — {horizon}")
                df_d = pd.DataFrame(drivers)
                df_d["shap_value"] = df_d["shap_value"].round(4)
                fig_d = px.bar(
                    df_d.head(10), x="shap_value", y="feature", orientation="h",
                    color="shap_value",
                    color_continuous_scale=["#f44336", "#ffffff", "#4caf50"],
                    color_continuous_midpoint=0,
                    title=f"SHAP — {horizon} (verde=alcista, rojo=bajista)",
                )
                fig_d.update_layout(height=320, margin=dict(t=40, b=0, l=0, r=0),
                                    showlegend=False, yaxis_title="", xaxis_title="SHAP")
                st.plotly_chart(fig_d, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — MARKET OVERVIEW
# ═════════════════════════════════════════════════════════════════════════════
with t_market:
    n_days = st.slider("Días a mostrar", 60, 2000, 500, key="mkt_days")
    df_plot = df_raw.tail(n_days).copy()

    show_sma20  = st.checkbox("SMA 20",  value=True)
    show_sma50  = st.checkbox("SMA 50",  value=True)
    show_sma200 = st.checkbox("SMA 200", value=True)
    show_boll   = st.checkbox("Bollinger Bands", value=False)
    show_sr     = st.checkbox("Soporte / Resistencia (20d)", value=True)

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.75, 0.25], vertical_spacing=0.04,
    )

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df_plot.index, open=df_plot["Open"], high=df_plot["High"],
        low=df_plot["Low"], close=df_plot["Close"],
        name="IBEX35", increasing_line_color="#26a69a",
        decreasing_line_color="#ef5350",
    ), row=1, col=1)

    # Overlays
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
            name="BB+", line=dict(color="#78909c", width=1, dash="dot"),
            fill=None), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_plot.index, y=bm - 2*bs,
            name="BB-", line=dict(color="#78909c", width=1, dash="dot"),
            fill="tonexty", fillcolor="rgba(120,144,156,0.10)"), row=1, col=1)

    if show_sr:
        res = df_plot["High"].rolling(20).max().shift(1)
        sup = df_plot["Low"].rolling(20).min().shift(1)
        fig.add_trace(go.Scatter(x=df_plot.index, y=res,
            name="Resistencia 20d", line=dict(color="#f44336", width=1, dash="dash")), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_plot.index, y=sup,
            name="Soporte 20d", line=dict(color="#4caf50", width=1, dash="dash")), row=1, col=1)

    # Volume bars
    vol_colors = ["#26a69a" if df_plot["Close"].iloc[i] >= df_plot["Open"].iloc[i]
                  else "#ef5350" for i in range(len(df_plot))]
    fig.add_trace(go.Bar(x=df_plot.index, y=df_plot["Volume"],
        name="Volumen", marker_color=vol_colors, opacity=0.7), row=2, col=1)

    fig.update_layout(
        height=620, xaxis_rangeslider_visible=False,
        margin=dict(t=20, b=20, l=0, r=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    fig.update_xaxes(showgrid=False)
    st.plotly_chart(fig, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 — TECHNICAL INDICATORS
# ═════════════════════════════════════════════════════════════════════════════
with t_indicators:
    st.subheader("Estado actual de indicadores técnicos")

    # ── State badge display ───────────────────────────────────────────────────
    STATE_COLOR = {
        "bullish":         "#4caf50",
        "bearish":         "#f44336",
        "neutral":         "#9e9e9e",
        "overbought":      "#ff9800",
        "oversold":        "#2196f3",
        "high_vol":        "#e91e63",
        "low_vol":         "#00bcd4",
        "near_resistance": "#ff5722",
        "near_support":    "#8bc34a",
        "middle":          "#9e9e9e",
    }
    STATE_LABEL_ES = {
        "bullish":         "Alcista",
        "bearish":         "Bajista",
        "neutral":         "Neutral",
        "overbought":      "Sobrecompra",
        "oversold":        "Sobreventa",
        "high_vol":        "Alta Volatilidad",
        "low_vol":         "Baja Volatilidad",
        "near_resistance": "Cerca Resistencia",
        "near_support":    "Cerca Soporte",
        "middle":          "En rango",
    }

    if not indicator_states:
        st.warning("Datos insuficientes para calcular indicadores (se necesitan >210 días).")
    else:
        categories = ["Tendencia", "Momentum", "Volatilidad", "Volumen", "Pivotes"]
        for cat in categories:
            cat_states = [s for s in indicator_states if s["category"] == cat]
            if not cat_states:
                continue

            st.markdown(f"### {cat}")
            n_cols = min(3, len(cat_states))
            cols   = st.columns(n_cols)
            for i, state in enumerate(cat_states):
                with cols[i % n_cols]:
                    color = STATE_COLOR.get(state["state"], "#9e9e9e")
                    label = STATE_LABEL_ES.get(state["state"], state["state"])
                    val   = f"{state['value']:.2f}" if state["value"] is not None else "N/A"
                    st.markdown(f"""
                    <div style="border-left: 4px solid {color}; padding: 8px 12px; margin-bottom:8px;
                                background: white; border-radius: 4px; box-shadow: 0 1px 3px rgba(0,0,0,0.1)">
                        <div style="font-size:12px; color:#666">{state['name']}</div>
                        <div style="font-size:20px; font-weight:bold; color:{color}">{label}</div>
                        <div style="font-size:12px; color:#333">{state['label']}</div>
                        <div style="font-size:11px; color:#999">valor: {val}</div>
                    </div>
                    """, unsafe_allow_html=True)
            st.divider()

    # ── Interactive feature chart ─────────────────────────────────────────────
    st.subheader("Explorador de indicadores")
    default_feats = ["rsi_14", "macd_hist", "stoch_k", "boll_pct_b",
                     "adx14", "cmf20", "hv_21d"]
    default_feats = [f for f in default_feats if f in feat_c][:3]
    selected = st.multiselect("Indicadores a visualizar", feat_c, default=default_feats)
    n_hist   = st.slider("Días de historia", 60, 1000, 252, key="ind_days")
    if selected:
        fig_ind = go.Figure()
        df_ind  = df_feat[selected].tail(n_hist)
        for col in selected:
            fig_ind.add_trace(go.Scatter(
                x=df_ind.index, y=df_ind[col], name=col, mode="lines",
            ))
        fig_ind.update_layout(
            height=350, margin=dict(t=10, b=0, l=0, r=0),
            legend=dict(orientation="h"),
        )
        st.plotly_chart(fig_ind, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 4 — EXPLAINABILITY
# ═════════════════════════════════════════════════════════════════════════════
with t_explain:
    st.subheader("Explicabilidad — SHAP")

    target_exp = st.selectbox(
        "Target",
        ["target_dir_1d", "target_dir_5d"],
        key="exp_target",
    )
    model_exp  = st.selectbox(
        "Modelo",
        ["xgboost", "lgbm", "random_forest", "ensemble", "logistic"],
        key="exp_model",
    )

    if st.button("Calcular SHAP (puede tardar ~30s)"):
        with st.spinner("Calculando valores SHAP..."):
            try:
                from src.models.ensemble import load_model
                from src.explainability.shap_analysis import compute_shap, shap_summary, shap_for_latest

                model = load_model(model_exp, target_exp)
                df_t  = df_feat.dropna(subset=[target_exp])
                X     = df_t[feat_c].values
                y     = df_t[target_exp].values

                shap_vals, f_names = compute_shap(model, X, feat_c, max_samples=500)
                df_imp = shap_summary(shap_vals, f_names, top_n=25)

                st.success(f"SHAP calculado sobre {len(shap_vals)} muestras")

                # Feature importance bar
                fig_imp = px.bar(
                    df_imp, x="mean_shap", y="feature", orientation="h",
                    title="Importancia media |SHAP| — top 25 features",
                    color="mean_shap",
                    color_continuous_scale="Blues",
                )
                fig_imp.update_layout(height=600, margin=dict(t=40, b=0, l=0, r=0),
                                      yaxis_title="", xaxis_title="mean |SHAP|")
                st.plotly_chart(fig_imp, use_container_width=True)

                # Latest prediction drivers
                st.subheader("Drivers de la última predicción")
                latest_x = df_t[feat_c].dropna().iloc[-1].values
                df_latest_shap = shap_for_latest(model, latest_x, feat_c)
                df_top = df_latest_shap.head(15)

                fig_wf = go.Figure(go.Bar(
                    x=df_top["shap_value"],
                    y=df_top["feature"],
                    orientation="h",
                    marker_color=["#4caf50" if v > 0 else "#f44336"
                                  for v in df_top["shap_value"]],
                    text=df_top["shap_value"].round(4),
                    textposition="outside",
                ))
                fig_wf.update_layout(
                    title="SHAP waterfall — predicción más reciente",
                    height=420, margin=dict(t=40, b=0, l=0, r=0),
                    xaxis_title="Impacto SHAP (verde=alcista, rojo=bajista)",
                    yaxis_title="",
                )
                st.plotly_chart(fig_wf, use_container_width=True)

                with st.expander("Tabla completa de features"):
                    st.dataframe(df_latest_shap.round(4))

            except FileNotFoundError as e:
                st.error(f"Modelo no encontrado: {e}\n\nEjecuta primero: `python train.py`")
            except ImportError as e:
                st.error(f"Dependencia faltante: {e}\n\nEjecuta: `pip install shap`")
            except Exception as e:
                st.error(f"Error: {e}")


# ═════════════════════════════════════════════════════════════════════════════
# TAB 5 — MODEL RESULTS
# ═════════════════════════════════════════════════════════════════════════════
with t_results:
    if not results:
        st.info("No hay resultados. Ejecuta: `python train.py`")
    else:
        target_r = st.selectbox("Target", list(results.keys()), key="res_target")
        r        = results[target_r]

        # ML metrics table
        ml_keys = ["accuracy", "balanced_accuracy", "f1", "f1_macro", "roc_auc", "pr_auc",
                   "mae", "rmse", "r2", "ic"]

        st.subheader("Métricas ML (media walk-forward)")
        rows = []
        for model, metrics in r.items():
            row = {"modelo": model}
            row.update({k: metrics[k] for k in ml_keys if k in metrics})
            rows.append(row)
        df_ml = pd.DataFrame(rows).set_index("modelo")
        if not df_ml.empty:
            st.dataframe(
                df_ml.style
                .format("{:.4f}")
                .highlight_max(axis=0, color="#c8e6c9")
                .highlight_min(axis=0, color="#ffcdd2"),
                use_container_width=True,
            )

        # Walk-forward predictions over time
        st.subheader("Predicciones walk-forward en el tiempo")
        model_sel = st.selectbox("Modelo", list(r.keys()), key="pred_model")
        pred_df   = load_predictions_df(target_r, model_sel)
        if pred_df is not None and "y_prob" in pred_df.columns:
            actual_dir = (pred_df["y_true"] == 1).astype(int)
            fig_pred = go.Figure()
            fig_pred.add_trace(go.Scatter(
                x=pred_df.index, y=pred_df["y_prob"],
                name="P(↑)", line=dict(color="#2196f3"),
            ))
            fig_pred.add_hline(y=0.5, line_dash="dash", line_color="gray")
            fig_pred.add_hline(y=get("backtest.min_confidence", 0.55),
                               line_dash="dot", line_color="green",
                               annotation_text="umbral confianza")
            # Mark correct / wrong predictions
            correct   = pred_df[pred_df["y_true"] == pred_df["y_pred"]]
            incorrect = pred_df[pred_df["y_true"] != pred_df["y_pred"]]
            fig_pred.add_trace(go.Scatter(
                x=correct.index, y=correct["y_prob"],
                mode="markers", name="Correcto",
                marker=dict(color="#4caf50", size=5, symbol="circle"),
            ))
            fig_pred.add_trace(go.Scatter(
                x=incorrect.index, y=incorrect["y_prob"],
                mode="markers", name="Incorrecto",
                marker=dict(color="#f44336", size=5, symbol="x"),
            ))
            fig_pred.update_layout(
                height=350, margin=dict(t=10, b=0, l=0, r=0),
                yaxis_range=[0, 1], yaxis_title="P(up)",
                legend=dict(orientation="h"),
            )
            st.plotly_chart(fig_pred, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 6 — BACKTEST
# ═════════════════════════════════════════════════════════════════════════════
with t_backtest:
    if not results:
        st.info("No hay resultados. Ejecuta: `python train.py`")
    else:
        target_bt = st.selectbox("Target", list(results.keys()), key="bt_target")
        r_bt      = results[target_bt]

        # Economic metrics table
        bt_metric_keys = ["bt_total_return", "bt_ann_return", "bt_ann_volatility",
                          "bt_sharpe", "bt_max_drawdown", "bt_calmar",
                          "bt_hit_ratio", "bt_n_trades",
                          "bt_bm_ann_return", "bt_excess_return"]

        st.subheader("Métricas económicas (media walk-forward)")
        rows_bt = []
        for model, metrics in r_bt.items():
            row = {"modelo": model}
            row.update({k.replace("bt_", ""): metrics[k] for k in bt_metric_keys if k in metrics})
            rows_bt.append(row)
        df_bt = pd.DataFrame(rows_bt).set_index("modelo")
        if not df_bt.empty:
            st.dataframe(
                df_bt.style.format("{:.4f}")
                .highlight_max(axis=0, subset=[c for c in df_bt.columns
                               if c in ["ann_return", "sharpe", "calmar", "hit_ratio",
                                        "excess_return"]], color="#c8e6c9")
                .highlight_min(axis=0, subset=[c for c in df_bt.columns
                               if c in ["max_drawdown", "ann_volatility"]], color="#c8e6c9"),
                use_container_width=True,
            )

        # Equity curves
        st.subheader("Curva de equity (walk-forward out-of-sample)")
        selected_models = st.multiselect(
            "Modelos a comparar", list(r_bt.keys()),
            default=list(r_bt.keys())[:3],
        )
        min_conf_bt = get("backtest.min_confidence", 0.55)
        tc_bps      = get("backtest.transaction_cost_bps", 10)

        fig_eq = go.Figure()
        has_bm = False

        for model_name in selected_models:
            pred_df = load_predictions_df(target_bt, model_name)
            if pred_df is None or "y_prob" not in pred_df.columns:
                continue
            if "target_ret_1d" in df_feat.columns:
                ret_series = df_feat["target_ret_1d"].reindex(pred_df.index).fillna(0)
            else:
                ret_series = pd.Series(0, index=pred_df.index)

            positions = np.where(pred_df["y_prob"] >= min_conf_bt, 1.0, 0.0)
            trades    = np.abs(np.diff(positions, prepend=0.0))
            strat_ret = positions * ret_series.values - trades * tc_bps / 10_000
            equity    = (1 + strat_ret).cumprod()

            fig_eq.add_trace(go.Scatter(
                x=pred_df.index, y=equity,
                name=model_name, mode="lines",
            ))

            if not has_bm:
                bm_equity = (1 + ret_series).cumprod()
                fig_eq.add_trace(go.Scatter(
                    x=pred_df.index, y=bm_equity,
                    name="Buy & Hold", mode="lines",
                    line=dict(color="gray", dash="dot"),
                ))
                has_bm = True

        if fig_eq.data:
            fig_eq.update_layout(
                height=400, yaxis_title="Valor cartera (base=1)",
                margin=dict(t=10, b=0, l=0, r=0),
                legend=dict(orientation="h"),
            )
            st.plotly_chart(fig_eq, use_container_width=True)
        else:
            st.info("Ejecuta `python train.py` para generar predicciones.")
