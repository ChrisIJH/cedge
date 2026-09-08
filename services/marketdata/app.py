"""
services/marketdata/app.py

Portfolio definitions and the return series they produce.

This is the service that turns a portfolio — a weight map, {ticker: weight} —
into something the risk service can score.

Run locally:
    python services/marketdata/app.py
"""
import os
import math
from typing import Optional

import numpy as np
import pandas as pd

from flask import Flask, request, jsonify

from cedge_core.db import ch_engine
from cedge_core.estimators import realized_beta
from cedge_core.marketdata.portfolios import list_portfolio, load_weights_asof
from cedge_core.marketdata.prices import load_prices_adjclose
from cedge_core.marketdata.returns import to_returns
from cedge_core.marketdata.weights import portfolio_returns

app = Flask(__name__)

DEFAULT_MARKET = "SPY"

class BadRequest(Exception):
    """A Client-side input problem."""

class NotFound(Exception):
    """The request is well-formed but the data not available"""

@app.errorhandler(BadRequest)
def _handle_bad_request(exc):
    return jsonify( {'error': str(exc)}), 400

@app.errorhandler(NotFound)
def _handle_not_found(exc):
    return jsonify( {'error': str(exc)}), 404

def _json_safe(values):
    """NaN -> None"""
    return [None if math.isnan(v) else float(v)
            for v in np.asarray(values, dtype=float)]


def _require_date(payload, key):
    value = payload.get(key)
    if not value:
        raise BadRequest(f"'{key}' is required (YYYY-MM-DD)")
    try:
        pd.Timestamp(value)
    except ValueError:
        raise BadRequest(f"'{key}' is not a date: {value!r}")
    return str(value)


def _require_weights(payload):
    weights = payload.get("weights")
    if not isinstance(weights, dict) or not weights:
        raise BadRequest("'weights' must be a non-empty object of {ticker: weight}")

    cleaned = {}
    for ticker, weight in weights.items():
        ticker = str(ticker).strip().upper()
        if not ticker:
            raise BadRequest("'weights' contains an empty ticker")
        try:
            weight = float(weight)
        except (TypeError, ValueError):
            raise BadRequest(f"weight for {ticker!r} is not a number: {weight!r}")
        if not math.isfinite(weight):
            raise BadRequest(f"weight for {ticker!r} must be finite, got {weight}")
        cleaned[ticker] = weight
    return cleaned



@app.route("/healthz", methods=["GET"])
def healthz():
    """Health Check"""
    return jsonify({"status": "ok"}), 200

@app.route("/api/portfolios", methods=['GET'])
def portfolios():
    """Every book in portfolio_weight, with its date coverage

    Column Keys:
        pf_name, experiment_id, branch, source
    
    Caller must pass all four back to /api/portfolio_weights
    """
    # Columns: pf_name, experiment_id, branch, source, start_date, end_date,n_dates, n_tickers.
    frame = list_portfolio(ch_engine())
    for column in ("start_date", "end_date"):
        frame[column] = frame[column].astype(str)
    return jsonify({"portfolios": frame.to_dict(orient="records")})



@app.route("/api/portfolio_weights", methods=["GET"])
def portfolio_weights():
    """The weight snapshot in force on as_of_date — not necessarily dated on it.
    """
    pf_name = request.args.get("pf_name")
    if not pf_name:
        raise BadRequest("'pf_name' is required")

    # These three are genuinely nullable for a live book, so an absent query
    # parameter means NULL here rather than "not specified".
    weights, trade_date = load_weights_asof(
        ch_engine(),
        pf_name,
        request.args.get("experiment_id"),
        request.args.get("branch"),
        request.args.get("source"),
        request.args.get("as_of_date"),
    )
    if not weights:
        raise NotFound(f"no weights found for pf_name={pf_name!r}")

    return jsonify({
        "pf_name": pf_name,
        "trade_date": str(trade_date),
        "weights": weights,
    })


@app.route("/api/portfolio_returns", methods=["POST"])
def returns():
    """Daily returns of a fixed-weight book, plus its exposures and market beta.

    FIXED WEIGHTS, NOT A REBALANCED BACKTEST. One weight map is applied across
    the whole window, so this answers "how would this book have behaved" — not
    "what would a strategy that rebalanced into it have earned". Applying
    today's positions to years of past returns is look-ahead in the strict
    sense; it is a legitimate framing, but only while it is labelled as one.

    The market series is loaded on its own full history so beta is estimated
    over every day both series exist, independent of the book's coverage.
    """
    payload = request.get_json(silent=True)
    if payload is None:
        raise BadRequest("request body must be JSON")

    weights = _require_weights(payload)
    start_date = _require_date(payload, "start_date")
    end_date = _require_date(payload, "end_date")
    market = str(payload.get("market", DEFAULT_MARKET)).strip().upper()

    tickers = list(weights)
    panel = load_prices_adjclose(
        ch_engine(), sorted(set(tickers) | {market}), start_date, end_date
    )

    missing = [t for t in tickers if t not in panel.columns]
    if missing:
        raise NotFound(
            f"no price data for {', '.join(missing)} between {start_date} and {end_date}"
        )

    # Returns are computed from this book's own columns. Taking them off the
    # combined panel would let one late-listing ticker — the market proxy
    # included — truncate the whole series through the shared dropna.
    book_returns = to_returns(panel[tickers], "simple")
    series, dates, exposure = portfolio_returns(book_returns, weights)

    if len(series) == 0:
        raise NotFound("no overlapping trading days for these tickers in this range")

    beta: Optional[dict] = None
    if market in panel.columns:
        market_returns = to_returns(panel[[market]], "simple")[market]
        aligned = market_returns.reindex(dates)
        if aligned.notna().any():
            beta = realized_beta(np.asarray(series, float),
                                 aligned.to_numpy(dtype=float))
            beta["market"] = market

    return jsonify({
        "dates": [str(d) for d in dates],
        "returns": _json_safe(series),
        "n_observations": int(len(series)),
        "gross": exposure["gross"],
        "net": exposure["net"],
        "beta": beta,
        "weighting": "fixed weights applied across the whole window; not a rebalanced backtest",
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8000")))