"""
apps/quant_ui/pages/1_Portfolio_Performance.py

Cumulative return chart for a long/short book, backed entirely by
services/portfolio_performance. 

The service takes a ticker list with a LONG/SHORT flag per ticker — legs
are equal-weighted, not weight-sized.
"""
from datetime import date

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from lib import api

st.set_page_config(page_title="Portfolio Performance", layout="wide")
st.title("Portfolio Performance")
st.caption(
    "Cumulative return of a long/short book, benchmarked against SPY — "
    "served entirely by the portfolio_performance service."
)

PRESETS = {
    "Momentum L/S": (["MTUM"], ["SPY"]),
    "Value L/S": (["VLUE"], ["SPY"]),
    "Quality L/S": (["QUAL"], ["SPY"]),
    "LowVol L/S": (["USMV"], ["SPY"]),
    "Size L/S": (["IWM"], ["SPY"]),
}

st.sidebar.header("Book")
preset_name = st.sidebar.selectbox("Preset", list(PRESETS) + ["Custom"])


if preset_name == "Custom":
    long_input = st.sidebar.text_input("Long tickers (comma-separated)", "AAPL")
    short_input = st.sidebar.text_input("Short tickers (comma-separated)", "MSFT")
    long_tickers = [t.strip().upper() for t in long_input.split(",") if t.strip()]
    short_tickers = [t.strip().upper() for t in short_input.split(",") if t.strip()]
else:
    long_tickers, short_tickers = PRESETS[preset_name]

if preset_name != "Custom":
    st.sidebar.caption(
        "ETF presets are currently blocked by a known bug."
    )


st.sidebar.header("Period")
start_date = st.sidebar.date_input("Start", value=date(2018, 1, 2))
end_date = st.sidebar.date_input("End", value=date(2024, 12, 31))

if not long_tickers or not short_tickers:
    st.warning("Provide at least one long ticker and one short ticker.")
    st.stop()

st.sidebar.divider()
run = st.sidebar.button("Run", type="primary", use_container_width=True)

if run:
    body = api.post("portfolio_performance", "/api/factor_portfolio_performance", {
        "tickers": long_tickers + short_tickers,
        "positions": (["LONG"] * len(long_tickers)) + (["SHORT"] * len(short_tickers)),
        "start_date": str(start_date),
        "end_date": str(end_date),
    })
    st.session_state["perf_body"] = body
    st.session_state["perf_long_tickers"] = long_tickers
    st.session_state["perf_short_tickers"] = short_tickers

if "perf_body" not in st.session_state:
    st.info("Set the book and period in the sidebar, then click **Run**.")
    st.stop()

body = st.session_state["perf_body"]
long_tickers = st.session_state["perf_long_tickers"]
short_tickers = st.session_state["perf_short_tickers"]

cum_returns = pd.DataFrame(body["cum_returns"])
cum_returns["date"] = pd.to_datetime(cum_returns["date"])
stats = pd.DataFrame(body["stats"])

st.caption(f"Long: {', '.join(long_tickers)}  ·  Short: {', '.join(short_tickers)}")

fig = go.Figure()
colors = {"L/S Portfolio": "#2f6fd0", "Long Leg": "#5a9c5a",
         "Short Leg": "#c9553d", "SPY": "#8a8f98"}
for column in cum_returns.columns:
    if column == "date":
        continue
    fig.add_trace(go.Scatter(
        x=cum_returns["date"], y=cum_returns[column], name=column, mode="lines",
        line=dict(width=2.2 if column == "L/S Portfolio" else 1.2,
                 color=colors.get(column), dash="dot" if column == "SPY" else "solid"),
        hovertemplate="%{y:.1%}<extra>" + column + "</extra>",
    ))
fig.update_layout(
    height=460, margin=dict(l=0, r=0, t=10, b=0),
    hovermode="x unified", yaxis_tickformat=".0%",
    legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0),
)
st.plotly_chart(fig, use_container_width=True)

st.subheader("Stats")
display_cols = ["Strategy", "Total Return", "Annual Return", "Annual Vol", "Sharpe", "Max Drawdown"]
st.dataframe(
    stats[display_cols].style.format({
        "Total Return": "{:.1%}", "Annual Return": "{:.1%}",
        "Annual Vol": "{:.1%}", "Sharpe": "{:.2f}", "Max Drawdown": "{:.1%}",
    }),
    use_container_width=True, hide_index=True,
)