"""
apps/quant_ui/2_VaR_Backtest.py


VaR/ES backtest for a long/short book: does the risk model's stated
confidence level actually hold up against what happened?

Pipeline: services/marketdata turns a portfolio into a return series;
services/risk scores that series against parametric and historic VaR at
several confidence levels in one call. This page holds no risk math of its
own — every number here comes from those two services over HTTP.
"""


from datetime import date

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from lib import api
from plotly.subplots import make_subplots

st.set_page_config(page_title="VaR Backtest", layout="wide")
st.title("VaR Backtest")

PRESETS = {
    "SPY (long-only)": {"SPY": 1.0},
    "Momentum L/S":    {"MTUM": 1.0, "SPY": -1.0},
    "Value L/S":       {"VLUE": 1.0, "SPY": -1.0},
    "Quality L/S":      {"QUAL": 1.0, "SPY": -1.0},
    "LowVol L/S":       {"USMV": 1.0, "SPY": -1.0},
    "Size L/S":         {"IWM": 1.0, "SPY": -1.0},
}

### sidebar

st.sidebar.header("Book")
source = st.sidebar.radio("Source", ["Preset", "From database"])

if source=='Preset':
    preset_name = st.sidebar.selectbox("Preset", list(PRESETS))
    weights = PRESETS[preset_name]
    book_label = preset_name
else:
    books = api.get("marketdata", "/api/portfolios")["portfolios"]
    if not books:
        st.warning("No portfolios found")
        st.stop()
    labels = [ f"{b['pf_name']} | exp={b['experiment_id']} | br={b['branch']} | {b['source']}"
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
    



st.sidebar.header("Period")
start_date = st.sidebar.date_input("Start", value=date(2015, 1, 1))
end_date = st.sidebar.date_input("End", value=date(2024, 12, 31))

st.sidebar.header("Backtest settings")
window = st.sidebar.slider("Rolling window", 60, 500, 252, step=10)
levels = st.sidebar.multiselect("Confidence levels", [0.95, 0.99], default=[0.95, 0.99],
                                format_func=lambda x: f"{x:.0%}")
methods = st.sidebar.multiselect("Methods", ["parametric", "historic"],
                                 default=["parametric", "historic"])
n_boot = st.sidebar.select_slider("Bootstrap draws (Acerbi-Székely)",
                                  [1000, 2000, 3000, 5000], value=3000)

if not levels or not methods:
    st.warning("Select at least one confidence level and one method.")
    st.stop()

alpha_levels = sorted({round(1.0 - c, 4) for c in levels}, reverse=True)

st.sidebar.divider()
run = st.sidebar.button("Run Backtest", type="primary", use_container_width=True)

if run:
    returns_body = api.post("marketdata", "/api/portfolio_returns", {
        "weights": weights, "start_date": str(start_date), "end_date": str(end_date),
    })

    if window >= returns_body["n_observations"]:
        st.warning(
            f"The window ({window}d) must be shorter than the sample "
            f"({returns_body['n_observations']} observations)."
        )
        st.stop()

    backtest_body = api.post("risk", "/api/full_backtest", {
        "returns": returns_body["returns"], "window": window,
        "levels": alpha_levels, "methods": methods, "n_boot": n_boot,
    })

    st.session_state["var_returns_body"] = returns_body
    st.session_state["var_backtest_body"] = backtest_body
    st.session_state["var_book_label"] = book_label

if "var_backtest_body" not in st.session_state:
    st.info("Set the book, period, and backtest settings in the sidebar, then click **Run Backtest**.")
    st.stop()

returns_body = st.session_state["var_returns_body"]
backtest_body = st.session_state["var_backtest_body"]
book_label = st.session_state["var_book_label"]

dates = pd.to_datetime(returns_body["dates"])
cells = backtest_body["cells"]

# summary

beta = returns_body["beta"]
col1, col2, col3, col4 = st.columns(4)
col1.metric("Book", book_label)
col2.metric("Observations", f"{returns_body['n_observations']:,}")
col3.metric("Gross / Net", f"{returns_body['gross']:.1f} / {returns_body['net']:+.1f}")
col4.metric(f"Beta vs {beta['market']}" if beta else "Beta", f"{beta['beta']:.2f}" if beta else "n/a")
st.caption(returns_body["weighting"])

# tabs

def cell_label(cell):
    return f"{cell['method']} {1 - cell['alpha']:.0%}"

tabs = st.tabs([cell_label(c) for c in cells])


for tab, cell in zip(tabs, cells):
    with tab:
        eval_dates = dates[cell["eval_idx"]]
        r_eval = pd.Series(cell["r_eval"], index=eval_dates)
        var = pd.Series(cell["var"], index=eval_dates)
        es = pd.Series(cell["es"], index=eval_dates)
        breach = pd.Series([bool(b) for b in cell["breaches"]], index=eval_dates)

        k = cell["kupiec"]
        c = cell["christoffersen"]
        a = cell["acerbi"]

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Breaches", f"{k['N']} / {k['T']}")
        m2.metric("Breach rate", f"{k['breach_rate']:.2%}",
                  delta=f"{k['breach_rate'] - k['expected_rate']:+.2%} vs expected")
        m3.metric("Kupiec", "Reject H\u2080 (Fail)" if k["reject"] else "Fail to reject H\u2080 (Pass)")
        m4.metric("Christoffersen", "Reject H\u2080 (Fail)" if c["reject"] else "Fail to reject H\u2080 (Pass)")

        # ---- 2-panel chart: returns/VaR/breach (top), rolling breach count (bottom)
        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True, row_heights=[0.62, 0.38],
            vertical_spacing=0.06,
            subplot_titles=["", "Breaches per rolling quarter — clustering"],
        )
        fig.add_trace(go.Scatter(x=eval_dates, y=r_eval, mode="lines", name="Return",
                                 line=dict(width=0.6, color="#8a8f98")), row=1, col=1)
        fig.add_trace(go.Scatter(x=eval_dates, y=-var, mode="lines", name="VaR",
                                 line=dict(width=1.4, color="#2f6fd0")), row=1, col=1)
        fig.add_trace(go.Scatter(x=eval_dates, y=-es, mode="lines", name="ES",
                                 line=dict(width=1.0, color="#2f6fd0", dash="dot")), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=eval_dates[breach], y=r_eval[breach], mode="markers",
            name=f"Breach ({int(breach.sum())})",
            marker=dict(color="#d1435b", size=6, symbol="x"),
        ), row=1, col=1)

        roll = breach.astype(int).rolling(63).sum()
        expected_roll = 63 * cell["alpha"]
        fig.add_trace(go.Scatter(x=roll.index, y=roll.values, mode="lines", fill="tozeroy",
                                 name="Breaches / 63d", line=dict(width=1.0, color="#d1435b")),
                      row=2, col=1)
        fig.add_hline(y=expected_roll, line_dash="dot", line_color="gray", row=2, col=1,
                     annotation_text=f"expected if independent ({expected_roll:.1f})")

        fig.update_layout(height=560, margin=dict(t=40, b=0, l=0, r=0), hovermode="x unified",
                         legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
        fig.update_yaxes(tickformat=".1%", row=1, col=1)
        st.plotly_chart(fig, use_container_width=True, key=f"chart_{cell_label(cell)}")

        # ---- clustering table
        st.subheader("Clustering — \u03c001 vs \u03c011")
        st.caption(
            "\u03c001 = P(breach | calm yesterday), \u03c011 = P(breach | breach yesterday). "
            "If breaches were independent these would be equal; a large multiple means "
            "breaches cluster — the model adapts too slowly to a shift in volatility."
        )
        pi01, pi11 = c["pi_01"], c["pi_11"]
        st.dataframe(pd.DataFrame([{
            "\u03c001": f"{pi01:.1%}", "\u03c011": f"{pi11:.1%}",
            "multiple (\u03c011/\u03c001)": f"{pi11 / pi01:.1f}\u00d7" if pi01 > 0 else "\u2014",
            "Christoffersen": "Reject H\u2080 (Fail)" if c["reject"] else "Fail to reject H\u2080 (Pass)",
        }]), use_container_width=True, hide_index=True)

        # ---- ES depth
        st.subheader("ES depth \u2014 predicted vs realized")
        ratio = cell["avg_real_loss"] / cell["avg_pred_es"] if cell["avg_pred_es"] else float("nan")
        st.dataframe(pd.DataFrame([{
            "predicted ES": f"{cell['avg_pred_es']:.2%}",
            "realized loss (avg breach)": f"{cell['avg_real_loss']:.2%}",
            "real / pred": f"{ratio:.2f}",
            "understated by": f"{ratio - 1:+.0%}",
            "Acerbi\u2013Sz\u00e9kely": "Reject H\u2080 (Fail)" if a["reject"] else "Fail to reject H\u2080 (Pass)",
        }]), use_container_width=True, hide_index=True)

        with st.expander("Reading these results"):
            st.markdown(f"""
**Kupiec** asks only whether breaches happen at the right *rate*: {k['N']} in
{k['T']} days observed vs {k['expected_rate']:.1%} expected.

**Christoffersen** adds *timing* on top of rate — a model can breach the right
number of times and still be wrong if those breaches arrive in bursts.

**Acerbi–Székely** tests the ES claim, not the VaR threshold: VaR says how
often losses cross the line, ES claims how bad the crossings are on average.

Both VaR and ES are **positive loss** numbers here: a VaR of 2% means a loss
of 2% or worse is expected on {cell['alpha']:.1%} of days.
""")