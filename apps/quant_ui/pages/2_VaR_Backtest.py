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

st.set_page_config(page_title="Cedge - VaR Backtest", layout="wide")
st.title("Cedge - VaR Backtest")

PRESETS = {
    "SPY (long-only)": {"SPY": 1.0},
    "Momentum L/S":    {"MTUM": 1.0, "SPY": -1.0},
    "Value L/S":       {"VLUE": 1.0, "SPY": -1.0},
    "Quality L/S":      {"QUAL": 1.0, "SPY": -1.0},
    "LowVol L/S":       {"USMV": 1.0, "SPY": -1.0},
    "Size L/S":         {"IWM": 1.0, "SPY": -1.0},
}

### URL parameters -> starting values of the sidebar widgets
# e.g. ?source=model&book=MOD_6040&start=2016-01-01&end=2026-11-06&level=0.99&method=historic&run=1


SOURCES = {"preset": "Preset", "model": "Model portfolio", "database": "From database"}


def url_choice(name, options, default):
    value = st.query_params.get(name)
    if value is None:
        return default
    if value not in options:
        st.sidebar.warning(f"Ignored URL parameter {name}={value!r}")
        return default
    return value


def url_choices(name, options, default, cast=str):
    value = st.query_params.get(name)
    if value is None:
        return default
    try:
        chosen = [cast(v) for v in value.split(",")]
    except ValueError:
        chosen = []
    if not chosen or any(c not in options for c in chosen):
        st.sidebar.warning(f"Ignored URL parameter {name}={value!r}")
        return default
    return chosen


def url_date(name, default):
    value = st.query_params.get(name)
    if value is None:
        return default
    try:
        return date.fromisoformat(value)
    except ValueError:
        st.sidebar.warning(f"Ignored URL parameter {name}={value!r}")
        return default


### sidebar

st.sidebar.header("Book")
source_key = url_choice("source", list(SOURCES), "preset")
source = st.sidebar.radio("Source", list(SOURCES.values()), index=list(SOURCES).index(source_key))

if source=='Preset':
    preset_names = list(PRESETS)
    preset_name = st.sidebar.selectbox(
        "Preset", preset_names, index=preset_names.index(url_choice("book", preset_names, preset_names[0])))
    weights = PRESETS[preset_name]
    book_label = preset_name
elif source == "Model portfolio":
    models = api.get("model_portfolio", "/api/model_portfolio/models")["models"]
    model_ids = sorted(models)
    model_id = st.sidebar.selectbox(
        "Model", model_ids, index=model_ids.index(url_choice("book", model_ids, model_ids[0])))
    weights = models[model_id]['weights']
    book_label = f"{model_id} v{models[model_id]['version']}"
    st.sidebar.caption(f"Weights from the model protfolio service ({len(weights)} ETFs)")
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

start_date = st.sidebar.date_input("Start", value=url_date("start", date(2015, 1, 1)))
end_date = st.sidebar.date_input("End", value=url_date("end", date(2026, 11, 3)))

st.sidebar.header("Backtest settings")
window = st.sidebar.slider("Rolling window", 60, 500, 252, step=10)
levels = st.sidebar.multiselect("Confidence levels", [0.95, 0.99],
                                default=url_choices("level", [0.95, 0.99], [0.95, 0.99], float),
                                format_func=lambda x: f"{x:.0%}")
METHODS = ["parametric", "historic", "fhs"]
methods = st.sidebar.multiselect("Methods", METHODS,
                                 default=url_choices("method", METHODS, ["parametric", "historic"]),
                                 help="fhs = filtered historical simulation (volatility-adjusted history)")

CHART_RANGES = {"full": "Full period", "12m": "Last 12 months"}
chart_range = st.sidebar.radio(
    "Chart range", list(CHART_RANGES.values()), horizontal=True,
    index=list(CHART_RANGES).index(url_choice("view", list(CHART_RANGES), "full")))

n_boot = st.sidebar.select_slider("Bootstrap draws (Acerbi-Székely)",
                                  [1000, 2000, 3000, 5000], value=3000)

if not levels or not methods:
    st.warning("Select at least one confidence level and one method.")
    st.stop()

alpha_levels = sorted({round(1.0 - c, 4) for c in levels}, reverse=True)

st.sidebar.divider()
run = st.sidebar.button("Run Backtest", type="primary", use_container_width=True)
if st.query_params.get("run") == "1" and "var_backtest_body" not in st.session_state:
    run = True

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

# price-data warnings (older marketdata services do not return them)
data_warnings = returns_body.get("data_warnings") or []
if data_warnings:
    shown = data_warnings[:10]
    lines = "\n".join(f"- {w['ticker']} on {w['date']}: {w['return']:+.1%} ({w['reason']})" for w in shown)
    more = f"\n- … and {len(data_warnings) - len(shown)} more" if len(data_warnings) > len(shown) else ""
    st.warning("**Possible price-data errors in this book.** Days like these are usually an "
               "unadjusted split, not a market move; results that include them can be wrong.\n"
               + lines + more)

# VaR charts can open on the last 12 months; the performance chart and every number use the whole sample.
view_end = dates[-1]
view_start = view_end - pd.DateOffset(months=12)
zoom = chart_range == CHART_RANGES["12m"]

# performance: cumulative return over drawdown (older marketdata services do not return them)
cumulative = returns_body.get("cumulative")
drawdown = returns_body.get("drawdown")
market_path = returns_body.get("market_path")
if cumulative is not None and drawdown is not None:
    cum = pd.Series(cumulative, index=dates)
    dd = pd.Series(drawdown, index=dates)
    fig_perf = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.6, 0.4],
                             vertical_spacing=0.06,
                             subplot_titles=["Cumulative return", "Drawdown — decline from the running peak"])
    fig_perf.add_trace(go.Scatter(x=dates, y=cum, mode="lines", name=book_label,
                                  line=dict(width=1.6, color="#2f6fd0")), row=1, col=1)
    fig_perf.add_trace(go.Scatter(x=dates, y=dd, mode="lines", fill="tozeroy", name="Drawdown",
                                  line=dict(width=1.0, color="#d1435b"), showlegend=False), row=2, col=1)
    if market_path is not None:
        fig_perf.add_trace(go.Scatter(x=dates, y=market_path["cumulative"], mode="lines",
                                      name=market_path["ticker"],
                                      line=dict(width=1.2, color="#8a8f98", dash="dash")), row=1, col=1)
        fig_perf.add_trace(go.Scatter(x=dates, y=market_path["drawdown"], mode="lines",
                                      name=f"{market_path['ticker']} drawdown", showlegend=False,
                                      line=dict(width=1.0, color="#8a8f98", dash="dash")), row=2, col=1)
    fig_perf.update_layout(height=460, margin=dict(t=40, b=0, l=0, r=0), hovermode="x unified",
                           legend=dict(orientation="h", yanchor="bottom", y=1.04, x=0))
    fig_perf.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig_perf, use_container_width=True)

    summary = (f"Total return {cum.iloc[-1]:+.1%} · deepest drawdown {dd.min():.1%} on "
               f"{dd.idxmin():%Y-%m-%d} · latest {dd.iloc[-1]:.1%}")
    if market_path is not None:
        m_cum = pd.Series(market_path["cumulative"], index=dates)
        m_dd = pd.Series(market_path["drawdown"], index=dates)
        summary += (f"  |  {market_path['ticker']}: {m_cum.iloc[-1]:+.1%} · "
                    f"{m_dd.min():.1%} · latest {m_dd.iloc[-1]:.1%}")
    st.caption(summary)
elif drawdown is not None:
    dd = pd.Series(drawdown, index=dates)
    fig_dd = go.Figure(go.Scatter(x=dates, y=dd, mode="lines", fill="tozeroy", name="Drawdown",
                                  line=dict(width=1.0, color="#d1435b")))
    fig_dd.update_layout(height=220, margin=dict(t=30, b=0, l=0, r=0),
                         title=dict(text="Drawdown — decline from the running peak", font=dict(size=14)))
    fig_dd.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig_dd, use_container_width=True)
    st.caption(f"Deepest: {dd.min():.1%} on {dd.idxmin():%Y-%m-%d} · latest: {dd.iloc[-1]:.1%}")

# tabs
METHOD_IN_WORDS = {
    "parametric": "normal-distribution estimate",
    "historic": f"history of the previous {window} days",
    "fhs": f"volatility-adjusted history of the previous {window} days (FHS)",
}

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

        days = round(1 / cell["alpha"])
        expected_n = k["expected_rate"] * k["T"]
        st.markdown(
            f"**1-in-{days}-day loss line** ({1 - cell['alpha']:.0%} VaR, "
            f"{METHOD_IN_WORDS[cell['method']]}): for {eval_dates[-1]:%Y-%m-%d} the line was a "
            f"**{var.iloc[-1]:.2%}** loss. Over {k['T']:,} days, losses went beyond the line on "
            f"**{k['N']}** days, where about **{expected_n:.0f}** were expected."
        )

        m1, m2 = st.columns(2)
        m1.metric("Breaches", f"{k['N']} / {k['T']}")
        m2.metric("Breach rate", f"{k['breach_rate']:.2%}",
                  delta=f"{k['breach_rate'] - k['expected_rate']:+.2%} vs expected")

        # ---- 2-panel chart: returns/VaR/breach (top), rolling breach count (bottom)
        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True, row_heights=[0.62, 0.38],
            vertical_spacing=0.06,
            subplot_titles=["", "Breaches per rolling quarter — clustering"],
        )
        conf = f"{1 - cell['alpha']:.0%}"
        fig.add_trace(go.Scatter(x=eval_dates, y=r_eval, mode="lines", name="Return",
                                 line=dict(width=0.6, color="#8a8f98")), row=1, col=1)
        fig.add_trace(go.Scatter(x=eval_dates, y=-var, mode="lines", name=f"VaR {conf}",
                                 line=dict(width=1.4, color="#2f6fd0")), row=1, col=1)
        fig.add_trace(go.Scatter(x=eval_dates, y=-es, mode="lines", name=f"ES {conf}",
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

        hits = eval_dates[breach.to_numpy()]
        if len(hits):
            last_hit = hits[-1]
            fig.add_annotation(x=last_hit, y=r_eval[last_hit], ax=-70, ay=30, row=1, col=1,
                               arrowcolor="#d1435b", font=dict(color="#d1435b"),
                               text=f"{last_hit:%b %d, %Y}: {r_eval[last_hit]:.2%}"
                                    f"<br>loss line {var[last_hit]:.2%}")

        recent = eval_dates >= view_start
        if recent.any():
            low = min(r_eval[recent].min(), -es[recent].max())
            fig.update_xaxes(range=[view_start, view_end])
            fig.update_yaxes(range=[low * 1.15, r_eval[recent].max() * 1.15], row=1, col=1)

        st.plotly_chart(fig, use_container_width=True, key=f"chart_{cell_label(cell)}")
        if zoom:
            st.caption("Showing the last 12 months — double-click the chart for the full period.")
            
        diagnostics = st.expander("Model diagnostics — statistical tests")
        m3, m4 = diagnostics.columns(2)
        m3.metric("Kupiec", "Reject H\u2080 (Fail)" if k["reject"] else "Fail to reject H\u2080 (Pass)")
        m4.metric("Christoffersen", "Reject H\u2080 (Fail)" if c["reject"] else "Fail to reject H\u2080 (Pass)")
        
        # ---- clustering table
        diagnostics.subheader("Clustering — \u03c001 vs \u03c011")
        diagnostics.caption(
            "\u03c001 = P(breach | calm yesterday), \u03c011 = P(breach | breach yesterday). "
            "If breaches were independent these would be equal; a large multiple means "
            "breaches cluster — the model adapts too slowly to a shift in volatility."
        )
        pi01, pi11 = c["pi_01"], c["pi_11"]
        diagnostics.dataframe(pd.DataFrame([{
            "\u03c001": f"{pi01:.1%}", "\u03c011": f"{pi11:.1%}",
            "multiple (\u03c011/\u03c001)": f"{pi11 / pi01:.1f}\u00d7" if pi01 > 0 else "\u2014",
            "Christoffersen": "Reject H\u2080 (Fail)" if c["reject"] else "Fail to reject H\u2080 (Pass)",
        }]), use_container_width=True, hide_index=True)

        # ---- ES depth
        diagnostics.subheader("ES depth \u2014 predicted vs realized")
        ratio = cell["avg_real_loss"] / cell["avg_pred_es"] if cell["avg_pred_es"] else float("nan")
        diagnostics.dataframe(pd.DataFrame([{
            "predicted ES": f"{cell['avg_pred_es']:.2%}",
            "realized loss (avg breach)": f"{cell['avg_real_loss']:.2%}",
            "real / pred": f"{ratio:.2f}",
            "understated by": f"{ratio - 1:+.0%}",
            "Acerbi\u2013Sz\u00e9kely": "Reject H\u2080 (Fail)" if a["reject"] else "Fail to reject H\u2080 (Pass)",
        }]), use_container_width=True, hide_index=True)

        diagnostics.subheader("Reading these results")
        diagnostics.markdown(f"""
**Kupiec** asks only whether breaches happen at the right *rate*: {k['N']} in
{k['T']} days observed vs {k['expected_rate']:.1%} expected.

**Christoffersen** adds *timing* on top of rate — a model can breach the right
number of times and still be wrong if those breaches arrive in bursts.

**Acerbi–Székely** tests the ES claim, not the VaR threshold: VaR says how
often losses cross the line, ES claims how bad the crossings are on average.

Both VaR and ES are **positive loss** numbers here: a VaR of 2% means a loss
of 2% or worse is expected on {cell['alpha']:.1%} of days.
""")