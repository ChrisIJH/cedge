"""
apps/quant_ui/4_Regime.py

Macro regime score, posterior, and policy — plus history and a backtest
against what CHResearch's production cron actually computed.

This page holds no regime math of its own — every number comes from
services/regime over HTTP.
"""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from lib import api

st.set_page_config(page_title="Regime", layout="wide")
st.title("Regime")

st.caption(
    "Macro z-score composite score -> fixed 3-Gaussian mixture posterior "
    "(risk_off / soft_patch / risk_on) -> policy (gross target, beta cap). "
    "Ported from CHResearch's ch_api/ch_metric.py::regime_dashboard."
)

### sidebar

st.sidebar.header("As-of")
as_of = st.sidebar.date_input("As-of date", value=pd.Timestamp.today().date())

st.sidebar.header("History / backtest window")
start_date = st.sidebar.date_input("Start", value=pd.Timestamp("2018-01-01").date())
end_date = st.sidebar.date_input("End", value=pd.Timestamp.today().date())
horizons = st.sidebar.multiselect("Forward horizons (days)", [1, 5, 20],
                                  default=[1, 5, 20])

if not horizons:
    st.warning("Select at least one horizon.")
    st.stop()

st.sidebar.divider()
run = st.sidebar.button("Load", type="primary", use_container_width=True)

if run:
    today_body = api.get("regime", "/api/regime", as_of=str(as_of))
    history_body = api.get("regime", "/api/regime/history",
                           start=str(start_date), end=str(end_date))
    backtest_body = api.get("regime", "/api/regime/backtest",
                            start=str(start_date), end=str(end_date),
                            horizons=",".join(str(h) for h in horizons))

    st.session_state["regime_today"] = today_body
    st.session_state["regime_history"] = history_body
    st.session_state["regime_backtest"] = backtest_body

if "regime_today" not in st.session_state:
    st.info("Set as-of date and history window in the sidebar, then click **Load**.")
    st.stop()

today = st.session_state["regime_today"]
history_body = st.session_state["regime_history"]
backtest_body = st.session_state["regime_backtest"]

### summary

posterior = today["posterior"]
dominant = max(posterior, key=posterior.get)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Score (0-100)", f"{today['score']:.1f}")
col2.metric("Dominant regime", dominant.replace("_", " "))
col3.metric("Gross target", f"{today['policy']['gross_target']:.1f}")
col4.metric("Beta cap", f"{today['policy']['beta_cap']:.2f}")

st.caption(
    f"As of {today['as_of']} — posterior: "
    f"risk_off {posterior['risk_off']:.1%} / "
    f"soft_patch {posterior['soft_patch']:.1%} / "
    f"risk_on {posterior['risk_on']:.1%}"
)

### history: score + posterior

st.subheader("History")

hist_df = pd.DataFrame(history_body["history"])
if hist_df.empty:
    st.warning("No history in this window.")
    st.stop()
hist_df["as_of_date"] = pd.to_datetime(hist_df["as_of_date"])
hist_df = hist_df.sort_values("as_of_date")

tab_score, tab_posterior, tab_zscores = st.tabs(["Score", "Posterior", "Z-score decomposition"])

with tab_score:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist_df["as_of_date"], y=hist_df["score"], mode="lines",
                             name="score", line=dict(width=1.2, color="#2f6fd0")))
    fig.add_hline(y=50, line_dash="dot", line_color="gray", annotation_text="neutral (50)")
    fig.update_layout(height=380, margin=dict(t=20, b=0, l=0, r=0), yaxis_range=[0, 100])
    st.plotly_chart(fig, use_container_width=True)

with tab_posterior:
    fig = go.Figure()
    colors = {"posterior_risk_off": "#d1435b", "posterior_soft_patch": "#c9a227",
             "posterior_risk_on": "#2f9e58"}
    for col, color in colors.items():
        if col in hist_df.columns:
            fig.add_trace(go.Scatter(x=hist_df["as_of_date"], y=hist_df[col],
                                     mode="lines", stackgroup="posterior",
                                     name=col.replace("posterior_", ""),
                                     line=dict(width=0.5, color=color)))
    fig.update_layout(height=380, margin=dict(t=20, b=0, l=0, r=0),
                      yaxis=dict(tickformat=".0%", range=[0, 1]),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
    st.plotly_chart(fig, use_container_width=True)

with tab_zscores:
    fig = go.Figure()
    for col in ("vix_z", "dxy_z", "ief_z", "credit_z"):
        if col in hist_df.columns:
            fig.add_trace(go.Scatter(x=hist_df["as_of_date"], y=hist_df[col],
                                     mode="lines", name=col, line=dict(width=1.0)))
    fig.add_hline(y=0, line_dash="dot", line_color="gray")
    fig.update_layout(height=380, margin=dict(t=20, b=0, l=0, r=0),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
    st.plotly_chart(fig, use_container_width=True)

### backtest

st.subheader("Backtest — forward return by regime label")
# st.warning(backtest_body["caveat"])

bucket_df = pd.DataFrame(backtest_body["bucket_stats"])
if bucket_df.empty:
    st.info("No bucket stats for this window.")
else:
    display_df = bucket_df.copy()
    for col in display_df.columns:
        if col.endswith("_mean"):
            display_df[col] = display_df[col].map(lambda x: f"{x:.2%}" if pd.notna(x) else "—")
        elif col.endswith("_count"):
            display_df[col] = display_df[col].map(lambda x: f"{int(x):,}" if pd.notna(x) else "—")
    st.dataframe(display_df, use_container_width=True, hide_index=True)

st.caption(
    f"{backtest_body['n_days']:,} trading days, "
    f"{backtest_body['start']} to {backtest_body['end']}, "
    f"score_name={backtest_body['score_name']}"
)

