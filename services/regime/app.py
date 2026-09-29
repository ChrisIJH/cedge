"""
services/regime/app.py

Macro regime score, posterior, and policy — plus history and a backtest
against what CHResearch's daily cron actually produced.

Run locally:
    python services/regime/app.py
"""
import os

import pandas as pd
from cedge_core.db import ch_engine
from cedge_core.marketdata.prices import load_prices_adjclose
from cedge_core.marketdata.returns import to_returns
from cedge_core.regime.backtest import bucket_stats_by_label, forward_returns
from cedge_core.regime.history import get_regime_history
from cedge_core.regime.model import compute_regime, prices_to_wide
from cedge_core.regime.prices import get_macro_prices
from flask import Flask, jsonify, request

app = Flask(__name__)

MACRO_TICKERS = ["^VIX", "UUP", "HYG", "IEF"]
WARMUP_DAYS = 400  # >= 252-day z-score window + buffer, matches CHResearch's cron


class BadRequest(Exception):
    """A client-side input problem."""


class NotFound(Exception):
    """The request is well-formed but the data not available."""


@app.errorhandler(BadRequest)
def _handle_bad_request(exc):
    return jsonify({"error": str(exc)}), 400


@app.errorhandler(NotFound)
def _handle_not_found(exc):
    return jsonify({"error": str(exc)}), 404


def _require_date_arg(key):
    value = request.args.get(key)
    if not value:
        raise BadRequest(f"'{key}' is required (YYYY-MM-DD)")
    try:
        pd.Timestamp(value)
    except ValueError:
        raise BadRequest(f"'{key}' is not a date: {value!r}")
    return str(value)


def _parse_horizons():
    raw = request.args.get("horizons", "1,5,20")
    try:
        horizons = [int(h) for h in raw.split(",") if h.strip()]
    except ValueError:
        raise BadRequest(f"'horizons' must be comma-separated integers, got {raw!r}")
    if not horizons:
        raise BadRequest("'horizons' must contain at least one integer")
    return horizons


@app.route("/healthz", methods=["GET"])
def healthz():
    """Health check."""
    return jsonify({"status": "ok"}), 200


@app.route("/api/regime", methods=["GET"])
def regime():
    """Regime score/posterior/policy for a single as-of date, computed
    fresh from prices (not read from CHResearch's regime_scores table)."""
    as_of = _require_date_arg("as_of")
    start = str(pd.Timestamp(as_of) - pd.Timedelta(days=WARMUP_DAYS))[:10]

    long_prices = get_macro_prices(MACRO_TICKERS, start, as_of)
    if long_prices.empty:
        raise NotFound(f"no macro price data between {start} and {as_of}")

    px_macro = prices_to_wide(long_prices)
    missing = [t for t in MACRO_TICKERS if t not in px_macro.columns]
    if missing:
        raise NotFound(f"no price data for {', '.join(missing)} between {start} and {as_of}")

    result = compute_regime(px_macro)

    return jsonify({
        "as_of": as_of,
        "score": result.score,
        "posterior": result.posterior,
        "policy": result.policy,
        "vix_z": result.vix_z,
        "dxy_z": result.dxy_z,
        "ief_z": result.ief_z,
        "credit_z": result.credit_z,
        "pca_factor": result.pca_factor,
    })


@app.route("/api/regime/history", methods=["GET"])
def history():
    """CHResearch's precomputed daily regime_scores, as actually written
    by the production cron -- not recomputed with cedge's core."""
    start = _require_date_arg("start")
    end = _require_date_arg("end")
    score_name = request.args.get("score_name", "macro_v1")

    df = get_regime_history(start, end, score_name)
    if df.empty:
        raise NotFound(f"no regime_scores for score_name={score_name!r} between {start} and {end}")

    df["as_of_date"] = df["as_of_date"].astype(str)
    return jsonify({"history": df.to_dict(orient="records")})


@app.route("/api/regime/backtest", methods=["GET"])
def backtest():
    """Forward-return stats bucketed by CHResearch's historical regime
    label. Validates what was ACTUALLY used in production -- see
    RegimeScoreRepository docstring on the credit sign-invariance bug
    fixed in cedge's core but not (yet) in CHResearch's ch_metric.py."""
    start = _require_date_arg("start")
    end = _require_date_arg("end")
    score_name = request.args.get("score_name", "macro_v1")
    horizons = _parse_horizons()

    regime_df = get_regime_history(start, end, score_name)
    if regime_df.empty:
        raise NotFound(f"no regime_scores for score_name={score_name!r} between {start} and {end}")
    regime_df = regime_df.set_index(pd.to_datetime(regime_df["as_of_date"]))

    spy_wide = load_prices_adjclose(ch_engine(), ["SPY"], start, end)
    if "SPY" not in spy_wide.columns:
        raise NotFound(f"no SPY price data between {start} and {end}")

    spy_ret = to_returns(spy_wide[["SPY"]], "simple")["SPY"]
    spy_ret.index = pd.to_datetime(spy_ret.index)

    joined = regime_df.join(spy_ret.rename("ret"), how="inner")
    if joined.empty:
        raise NotFound("no overlapping trading days between regime_scores and SPY prices")

    fwd_df = forward_returns(joined["ret"], horizons=horizons)
    stats = bucket_stats_by_label(joined, fwd_df)

    return jsonify({
        "start": start,
        "end": end,
        "score_name": score_name,
        "horizons": horizons,
        "n_days": len(joined),
        "bucket_stats": stats.reset_index().rename(columns={"index": "regime_label"}).to_dict(orient="records"),
        "caveat": ("Backtests CHResearch's actual production regime_scores, "
                  "including the credit sign-invariance bug fixed in cedge's "
                  "core (PR #26) but not in CHResearch's own ch_metric.py."),
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8004")))