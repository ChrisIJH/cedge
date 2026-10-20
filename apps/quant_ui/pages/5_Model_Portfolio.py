"""
apps/quant_ui/pages/5_Model_Portfolio.py

Model portfolio lab: one rebalance run over a seeded synthetic book at real
closes — reconciliation, drift before/after, band calibration, the exception
queue, account size effects, attribution, and one account's before/after.

Every number comes from services/model_portfolio over HTTP.
"""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from lib import api

st.set_page_config(page_title="Model Portfolio", layout="wide")
st.title("Model Portfolio")
st.caption(
    "Model weights → whole-share orders across many accounts. Learning prototype: "
    "1,000 seeded synthetic accounts plus ten hand-made edge cases, valued at real closes."
)

SERVICE = "model_portfolio"
STATUS_COLORS = {"TRADED": "#2c7a4b", "SKIPPED_IN_BAND": "#8a959e",
                 "EXC_TOO_SMALL": "#b07a1a", "EXC_INPUT": "#b42b1e", "EXC_INVARIANT": "#6b1d8f"}

### sidebar

st.sidebar.header("Run")
asof = st.sidebar.date_input("As-of (trade date)", value=pd.Timestamp("2026-10-16").date())
band = st.sidebar.select_slider("Band", options=[0.01, 0.02, 0.03, 0.05], value=0.05,
                                format_func=lambda b: f"{b:.0%}")
cash_buffer = st.sidebar.select_slider("Cash buffer", options=[0.0, 0.01, 0.02, 0.03, 0.05],
                                       value=0.02, format_func=lambda b: f"{b:.0%}")
st.sidebar.header("Synthetic book")
n = st.sidebar.number_input("Accounts", min_value=100, max_value=5000, value=1000, step=100)
seed = st.sidebar.number_input("Seed", min_value=0, value=20261014, step=1)
st.sidebar.divider()
load = st.sidebar.button("Run", type="primary", use_container_width=True)

params = {"asof": str(asof), "band": band, "cash_buffer": cash_buffer, "n": int(n), "seed": int(seed)}

if load:
    with st.spinner("Running the book (attribution takes a few seconds the first time)…"):
        st.session_state["mp"] = {
            "params": params,
            "models": api.get(SERVICE, "/api/model_portfolio/models"),
            "run": api.get(SERVICE, "/api/model_portfolio/run", **params),
            "bands": api.get(SERVICE, "/api/model_portfolio/bands", **params),
            "attr": api.get(SERVICE, "/api/model_portfolio/attribution", **params),
        }

if "mp" not in st.session_state:
    st.info("Pick an as-of date and rules in the sidebar, then click **Run**.")
    st.stop()

mp = st.session_state["mp"]
run = mp["run"]

### 1. run summary

if run["status"] == "ABORTED":
    st.error(f"Run {run['run_id']} aborted: {run['abort_reason']}")
    st.stop()

c = run["counts"]
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Accounts", f"{c['n_target']:,}")
col2.metric("Traded", f"{c['n_processed']:,}")
col3.metric("Exceptions", f"{c['n_exception']:,}")
col4.metric("Orders", f"{run['n_orders']:,}")
col5.metric("Book turnover", f"{run['book_turnover']:.2%}")

recon_ok = run["status"] == "OK"
(st.success if recon_ok else st.error)(
    f"Reconciliation: {run['reconciliation']}  ·  rerun after fills: "
    f"{run['rerun_n_processed']} accounts trade  ·  run_id {run['run_id']}"
)

accounts = pd.DataFrame(run["accounts"])
trades = pd.DataFrame(run["trades"])

### 2. models

with st.expander("Models and benchmarks", expanded=False):
    rows = []
    for mid, m in mp["models"]["models"].items():
        for t, w in m["weights"].items():
            rows.append({"portfolio": mid, "ticker": t, "weight": w})
        for t, w in m["benchmark"]["weights"].items():
            rows.append({"portfolio": m["benchmark"]["id"], "ticker": t, "weight": w})
    wdf = pd.DataFrame(rows).pivot(index="ticker", columns="portfolio", values="weight").fillna(0.0)
    fig = go.Figure([go.Bar(name=col, x=wdf.index, y=wdf[col]) for col in wdf.columns])
    fig.update_layout(barmode="group", height=300, margin=dict(t=10, b=0, l=0, r=0),
                      yaxis_tickformat=".0%", legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig, use_container_width=True)
    st.caption(mp["models"]["benchmark_note"])

### 3. drift before / after

st.subheader("Drift before and after")
valid = accounts[accounts["status"] != "EXC_INPUT"]
fig = go.Figure()
fig.add_trace(go.Histogram(x=valid["drift_before"], name="before", opacity=0.6,
                           xbins=dict(size=0.0025), marker_color="#8a959e"))
traded = valid[valid["status"] == "TRADED"]
fig.add_trace(go.Histogram(x=traded["drift_after"], name="after (traded accounts)", opacity=0.75,
                           xbins=dict(size=0.0025), marker_color="#0e6a70"))
fig.add_vline(x=band, line_dash="dash", line_color="#b42b1e",
              annotation_text=f"band {band:.0%}", annotation_position="top")
fig.update_layout(barmode="overlay", height=320, margin=dict(t=30, b=0, l=0, r=0),
                  xaxis_title="max |weight − target| per account", xaxis_tickformat=".0%",
                  yaxis_title="accounts", legend=dict(orientation="h", y=1.12))
st.plotly_chart(fig, use_container_width=True)

### 4. band calibration

st.subheader("Band calibration")
bands = pd.DataFrame(mp["bands"]["bands"])
left, right = st.columns([3, 2])
with left:
    fig = go.Figure()
    fig.add_trace(go.Bar(x=[f"{b:.0%}" for b in bands["band"]], y=bands["n_traded"],
                         name="accounts traded", marker_color="#0e6a70"))
    fig.add_trace(go.Scatter(x=[f"{b:.0%}" for b in bands["band"]], y=bands["turnover_book"],
                             name="book turnover", yaxis="y2", mode="lines+markers",
                             line=dict(color="#b07a1a", width=2)))
    fig.update_layout(height=300, margin=dict(t=10, b=0, l=0, r=0),
                      yaxis=dict(title="accounts traded"),
                      yaxis2=dict(title="turnover", overlaying="y", side="right", tickformat=".1%"),
                      legend=dict(orientation="h", y=1.12))
    st.plotly_chart(fig, use_container_width=True)
with right:
    show = bands.rename(columns={"n_triggered": "out of band", "n_traded": "traded",
                                 "n_too_small": "too small", "n_orders": "orders"})
    show["band"] = show["band"].map(lambda b: f"{b:.0%}")
    show["turnover"] = show["turnover_book"].map(lambda x: f"{x:.2%}")
    st.dataframe(show[["band", "out of band", "traded", "too small", "orders", "turnover"]],
                 hide_index=True, use_container_width=True)

### 5. exception queue

st.subheader("Exception queue")
exc = accounts[accounts["status"].str.startswith("EXC")].copy()
if exc.empty:
    st.success("No exceptions in this run.")
else:
    exc["total_value"] = exc["total_value"].map(lambda v: f"${v:,.0f}" if pd.notna(v) else "—")
    st.dataframe(exc[["account_id", "model_id", "status", "reason", "flag_codes", "total_value"]],
                 hide_index=True, use_container_width=True)
    st.caption("EXC_INPUT: the account's own inputs are bad (never traded). "
               "EXC_TOO_SMALL: out of band, but no whole-share order is possible.")

### 6. account size

st.subheader("Account size vs drift left after trading")
fig = go.Figure()
for status, color in STATUS_COLORS.items():
    sub = valid[valid["status"] == status]
    if sub.empty:
        continue
    fig.add_trace(go.Scatter(x=sub["total_value"], y=sub["drift_after"], mode="markers",
                             name=status, marker=dict(size=6, color=color, opacity=0.7),
                             text=sub["account_id"]))
for x, label in ((322, "$322: no whole share of any fund"), (1_899, "$1,899: one IVV share (6040)")):
    fig.add_vline(x=x, line_dash="dot", line_color="#8a959e", annotation_text=label,
                  annotation_position="top right")
fig.add_hline(y=band, line_dash="dash", line_color="#b42b1e")
fig.update_layout(height=380, margin=dict(t=30, b=0, l=0, r=0), xaxis_type="log",
                  xaxis_title="account value ($, log)", yaxis_title="max drift after",
                  yaxis_tickformat=".0%", legend=dict(orientation="h", y=1.1))
st.plotly_chart(fig, use_container_width=True)

### 7. attribution

st.subheader("Where the return gap came from")
attr = mp["attr"]
st.caption(attr["note"])
by_model = pd.DataFrame(attr["by_model_bp"]).set_index("model_id")
cols = ["benchmark", "model", "account", "active", "allocation", "selection", "interaction",
        "cash", "rounding", "drift"]
st.markdown("**Median by model (bp)** — account − benchmark = allocation + selection + "
            "interaction (decision) + cash + rounding (implementation)")
st.dataframe(by_model[cols], use_container_width=True)

left, right = st.columns([2, 3])
with left:
    st.markdown("**Implementation effect by account size (median bp)**")
    size = pd.DataFrame(attr["by_size"])
    size["cash_start"] = size["cash_start"].map(lambda x: f"{x:.1%}")
    st.dataframe(size.round(1), hide_index=True, use_container_width=True)
with right:
    per = pd.DataFrame(attr["accounts"])
    per["implementation_bp"] = (per["cash"] + per["rounding"]) * 1e4
    fig = go.Figure(go.Scatter(x=per["value_start"], y=per["implementation_bp"], mode="markers",
                               marker=dict(size=5, color="#0e6a70", opacity=0.6),
                               text=per["account_id"]))
    fig.update_layout(height=320, margin=dict(t=10, b=0, l=0, r=0), xaxis_type="log",
                      xaxis_title="account value at last rebalance ($, log)",
                      yaxis_title="cash + rounding (bp)")
    st.plotly_chart(fig, use_container_width=True)

### 8. account detail

st.subheader("Account detail")
ids = accounts["account_id"].tolist()
default = ids.index("E02_LEGACY") if "E02_LEGACY" in ids else 0
account_id = st.selectbox("Account", ids, index=default)
detail = api.get(SERVICE, "/api/model_portfolio/account", account_id=account_id, **mp["params"])
if detail["status"] == "EXC_INPUT":
    st.error(f"{account_id}: excluded before trading ({detail['reason']}). "
             f"Holdings {detail['holdings']}, cash ${detail['cash']:,.2f}")
else:
    a, b, c3 = st.columns(3)
    a.metric("Status", detail["status"])
    b.metric("Value", f"${detail['total_value']:,.0f}")
    c3.metric("Max drift", f"{detail['max_drift_before']:.2%} → {detail['max_drift_after']:.2%}")
    if detail["flags"]:
        st.caption("Flags: " + ", ".join(detail["flags"]))
    d = pd.DataFrame(detail["rows"])
    for col in ("weight_before", "target", "weight_after", "drift_before", "drift_after"):
        d[col] = d[col].map(lambda x: f"{x:.2%}")
    st.dataframe(d, hide_index=True, use_container_width=True)
