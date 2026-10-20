"""
services/portfolio_performance/app.py

Long/short book performance over HTTP: cumulative returns per leg plus
annualized stats, from cedge_core.portfolio.performance.build_performance().

Extracted from the CHRiskAnalytics app_prod.py monolith (3,900 lines,
pulling in pymc/arviz/scikit-learn/MongoDB) so this service carries only
the dependencies it actually needs.

Run locally (from the repo root):
    python -m services.portfolio_performance.app
"""
import os

import pandas as pd
from cedge_core.portfolio.performance import build_performance
from flask import Flask, jsonify

from services.common import (
    BadRequest,
    NotFound,
    json_safe_dict,
    register_error_handlers,
    require_date,
    require_json,
    require_tickers,
)

app = Flask(__name__)
register_error_handlers(app)

POSITIONS = ("LONG", "SHORT")


def _require_positions(payload, n_tickers):
    positions = payload.get("positions")
    if not isinstance(positions, list) or len(positions) != n_tickers:
        raise BadRequest(f"'positions' must be a list with one entry per ticker ({n_tickers})")
    cleaned = [str(p).strip().upper() for p in positions]
    bad = sorted(set(cleaned) - set(POSITIONS))
    if bad:
        raise BadRequest(f"'positions' entries must be LONG or SHORT, got {bad}")
    return cleaned


@app.route("/api/factor_portfolio_performance", methods=["POST"])
def factor_portfolio_performance():
    payload = require_json()
    tickers = require_tickers(payload)
    positions = _require_positions(payload, len(tickers))
    start_date = require_date(payload, "start_date")
    end_date = require_date(payload, "end_date")

    weights_df = pd.DataFrame({"ticker": tickers, "position": positions})
    result = build_performance(weights_df, start_date, end_date)
    if result is None:
        raise NotFound(f"no price data for {', '.join(tickers)} between {start_date} and {end_date}")

    cum_returns = result["cum_returns"].reset_index().rename(columns={"index": "date"})
    cum_returns["date"] = cum_returns["date"].astype(str)

    return jsonify({
        "cum_returns": [json_safe_dict(row) for row in cum_returns.to_dict(orient="records")],
        "stats": [json_safe_dict(row) for row in result["stats"]],
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
