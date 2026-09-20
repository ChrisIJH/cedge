"""
apps/quant_ui/3_Portfolio_Optimization.py

Given a book's current weights, propose a new allocation that minimizes
risk subject to gross/net/turnover/per-factor constraints.

Pipeline: services/optimization builds a covariance matrix (either sample
covariance from services/marketdata's return history, or a Barra-style
factor risk model from factor_betas_bayes/factor_covariance/
factor_resid_var) and solves a mean-variance QP via cedge_core.optimization.
This page holds no optimization math of its own — every number here comes
from that service over HTTP.
"""

from datetime import date

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from lib import api

st.set_page_config(page_title="Portfolio Optimization", layout="wide")
st.title("Portfolio Optimization")

PRESETS = {
    "SPY (long-only)": {"SPY": 1.0},
    "Momentum L/S":    {"MTUM": 1.0, "SPY": -1.0},
    "Value L/S":       {"VLUE": 1.0, "SPY": -1.0},
    "Quality L/S":      {"QUAL": 1.0, "SPY": -1.0},
    "LowVol L/S":       {"USMV": 1.0, "SPY": -1.0},
    "Size L/S":         {"IWM": 1.0, "SPY": -1.0},
}

### sidebar — book

st.sidebar.header("Book")
source = st.sidebar.radio("Source", ["Preset", "From database"])

if source == "Preset":
    preset_name = st.sidebar.selectbox("Preset", list(PRESETS))
    weights = PRESETS[preset_name]
    book_label = preset_name
else:
    books = api.get("marketdata", "/api/portfolios")["portfolios"]
    if not books:
        st.warning("No portfolios found")
        st.stop()
    labels = [f"{b['pf_name']} | exp={b['experiment_id']} | br={b['branch']} | {b['source']}"
              for b in books]
    choice = st.sidebar.selectbox("Book", range(len(books)), format_func=lambda i: labels[i])
    chosen = books[choice]
    as_of = st.sidebar.date_input(
        "As-of date", value=pd.Timestamp(chosen["end_date"]).date(),
        min_value=pd.Timestamp(chosen["start_date"]).date(),
        max_value=pd.Timestamp(chosen["end_date"]).date(),
    )
    snapshot = api.get("marketdata", "/api/portfolio_weights",
                       pf_name=chosen["pf_name"], experiment_id=chosen["experiment_id"],
                       branch=chosen["branch"], source=chosen["source"], as_of_date=str(as_of))
    weights = snapshot["weights"]
    book_label = f"{chosen['pf_name']} @ {snapshot['trade_date']}"
    st.sidebar.caption(f"Resolved snapshot: {snapshot['trade_date']} ({len(weights)} positions)")

### sidebar — covariance method

st.sidebar.header("Covariance")
cov_method = st.sidebar.radio(
    "Method", ["sample", "factor_model"],
    format_func=lambda m: "Sample covariance" if m == "sample" else "Factor risk model",
)

extra_params = {"cov_method": cov_method}

if cov_method == "sample":
    st.sidebar.caption(
        "Raw empirical covariance from daily returns. Best for small "
        "universes — degenerates for wide ones (see caption below)."
    )
    start_date = st.sidebar.date_input("Return window start", value=date(2023, 1, 1))
    end_date = st.sidebar.date_input("Return window end", value=date(2024, 12, 31))
    extra_params["start_date"] = str(start_date)
    extra_params["end_date"] = str(end_date)
else:
    st.sidebar.caption(
        "Sigma = B \u00b7 Sigma_f \u00b7 B\u1d40 + D — Bayesian factor betas, "
        "factor covariance, and idiosyncratic variance from CHResearch's "
        "factor_betas_bayes/factor_covariance/factor_resid_var tables. "
        "More stable across wide universes."
    )
    asof_date = st.sidebar.date_input("As-of date", value=date(2026, 9, 18))
    extra_params["asof_date"] = str(asof_date)

### sidebar — constraints

st.sidebar.header("Constraints")
gross_target = st.sidebar.slider("Gross target", 0.5, 4.0, 2.0, step=0.1)
net_target = st.sidebar.slider("Net target", -1.0, 1.0, 0.0, step=0.1)
w_bounds = st.sidebar.slider("Per-position bound", 0.05, 1.0, 0.5, step=0.05)
lambda_turnover = st.sidebar.slider("Turnover penalty (\u03bb)", 0.0, 5.0, 1.0, step=0.1)

extra_params.update({
    "gross_target": gross_target,
    "net_target": net_target,
    "w_bounds": w_bounds,
    "lambda_turnover": lambda_turnover,
})

#####
#####

st.sidebar.divider()
run = st.sidebar.button("Run Optimization", type="primary", use_container_width=True)

if run:
    body = {"strategy": "mean_variance", "weights": weights, **extra_params}
    result = api.post("optimization", "/api/optimize", body)
    st.session_state["opt_result"] = result
    st.session_state["opt_weights_before"] = weights
    st.session_state["opt_book_label"] = book_label
    st.session_state["opt_gross_target"] = gross_target
    st.session_state["opt_net_target"] = net_target
    st.session_state["opt_lambda_turnover"] = lambda_turnover

if "opt_result" not in st.session_state:
    st.info("Set the book and constraints in the sidebar, then click **Run Optimization**.")
    st.stop()

result = st.session_state["opt_result"]
weights = st.session_state["opt_weights_before"]
book_label = st.session_state["opt_book_label"]
gross_target = st.session_state["opt_gross_target"]
net_target = st.session_state["opt_net_target"]
lambda_turnover = st.session_state["opt_lambda_turnover"]

if result.get("dropped_tickers"):
    st.warning(
        f"{len(result['dropped_tickers'])} ticker(s) excluded — no factor data available: "
        f"{', '.join(result['dropped_tickers'])}"
    )

# summary

col1, col2, col3, col4 = st.columns(4)
col1.metric("Book", book_label)
col2.metric("Status", result["status"])
col3.metric("Solver", result["solver"])
col4.metric("Risk (variance)", f"{result['risk']:.2e}")

col5, col6, col7 = st.columns(3)
col5.metric("Gross realized", f"{result['gross_realized']:.2f}", delta=f"target {gross_target:.2f}")
col6.metric("Net realized", f"{result['net_realized']:+.2f}", delta=f"target {net_target:+.2f}")
col7.metric("Turnover", f"{result['turnover']:.2f}")

# before/after weights chart

st.subheader("Weights: before \u2192 after")
tickers = sorted(set(weights) | set(result["weights"]))
before = [weights.get(t, 0.0) for t in tickers]
after = [result["weights"].get(t, 0.0) for t in tickers]

fig = go.Figure()
fig.add_trace(go.Bar(x=tickers, y=before, name="Before", marker_color="#8a8f98"))
fig.add_trace(go.Bar(x=tickers, y=after, name="After", marker_color="#2f6fd0"))
fig.update_layout(barmode="group", height=420, margin=dict(t=20, b=0, l=0, r=0),
                  yaxis_tickformat=".1%", hovermode="x unified",
                  legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
st.plotly_chart(fig, use_container_width=True)

# diagnostics table

st.subheader("Diagnostics")
st.dataframe(pd.DataFrame([{
    "cov_method": result["cov_method"],
    "status": result["status"],
    "solver": result["solver"],
    "risk": f"{result['risk']:.4e}",
    "turnover": f"{result['turnover']:.4f}",
    "gross_realized": f"{result['gross_realized']:.4f}",
    "net_realized": f"{result['net_realized']:.4f}",
}]), use_container_width=True, hide_index=True)

with st.expander("Reading these results"):
    st.markdown(f"""
**Gross realized** vs **target** shows how tightly the leverage constraint
bound — if they're not close, another constraint (turnover, per-position
bound) is likely binding first.

**Turnover** is the L1 distance between before/after weights, penalized in
the objective by \u03bb={lambda_turnover:.1f} — raising \u03bb pulls the
solution back toward the current book at the cost of leaving risk/return
on the table.

**{result['cov_method']}** covariance: {"raw empirical covariance — reliable "
"here because this book is small; would need far more history than exists "
"to stay well-conditioned on a wide universe." if result['cov_method'] == "sample"
else "Barra-style factor decomposition — stays well-conditioned regardless "
"of universe size, since risk is estimated through ~6 factors rather than "
"per-asset pairwise covariances."}
""")