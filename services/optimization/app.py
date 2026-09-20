"""
ervices/optimization/app.py

Portfolio optimization — allocate capital across given holdings to
minimize risk subject to constraints like gross leverage, market beta cap,
turnover cost, or alpha maximization.
"""
import os

import numpy as np
import pandas as pd
from cedge_core.db import ch_engine
from cedge_core.marketdata.factor_risk import SqlFactorRiskRepository
from cedge_core.marketdata.prices import load_prices_adjclose
from cedge_core.marketdata.returns import to_returns
from cedge_core.optimization.covariance import (
    factor_model_covariance,
    sample_covariance,
)
from cedge_core.optimization.factory import solve
from flask import Flask, jsonify, request

app = Flask(__name__)

_factor_repo = SqlFactorRiskRepository()
_marketdata_api = os.getenv("CEDGE_MARKETDATA_API_URL", "http://localhost:8002")


class BadRequest(Exception):
    """Client-side input problem."""


class NotFound(Exception):
    """The data is not available."""


class ServiceError(Exception):
    """Service-side failure (e.g., optimization failed)."""


@app.errorhandler(BadRequest)
def _handle_bad_request(exc):
    return jsonify({"error": str(exc)}), 400


@app.errorhandler(NotFound)
def _handle_not_found(exc):
    return jsonify({"error": str(exc)}), 404


@app.errorhandler(ServiceError)
def _handle_service_error(exc):
    return jsonify({"error": str(exc)}), 500


def _json_safe(values):
    """NaN → None for JSON serialization."""
    import math
    return [None if (isinstance(v, float) and math.isnan(v)) else float(v)
            for v in np.asarray(values, dtype=float)]


def _require_dict(payload, key):
    value = payload.get(key)
    if not isinstance(value, dict) or not value:
        raise BadRequest(f"'{key}' must be a non-empty object")
    return value


def _require_strategy(payload):
    strategy = payload.get("strategy")
    if not strategy:
        raise BadRequest("'strategy' is required (e.g. 'mean_variance')")
    return str(strategy)


def _require_cov_method(payload):
    method = payload.get("cov_method")
    if method not in ("sample", "factor_model"):
        raise BadRequest(
            f"'cov_method' must be 'sample' or 'factor_model', got {method!r}"
        )
    return method


def _require_weights(payload):
    weights = _require_dict(payload, "weights")
    cleaned = {}
    for ticker, weight in weights.items():
        ticker = str(ticker).strip().upper()
        if not ticker:
            raise BadRequest("'weights' contains an empty ticker")
        try:
            weight = float(weight)
        except (TypeError, ValueError):
            raise BadRequest(f"weight for {ticker!r} is not a number: {weight!r}")
        if not np.isfinite(weight):
            raise BadRequest(f"weight for {ticker!r} must be finite, got {weight}")
        cleaned[ticker] = weight
    return cleaned


def _require_date(payload, key):
    value = payload.get(key)
    if not value:
        raise BadRequest(f"'{key}' is required (YYYY-MM-DD)")
    try:
        pd.Timestamp(value)
    except ValueError:
        raise BadRequest(f"'{key}' is not a valid date: {value!r}")
    return str(value)



def _require_tickers(payload):
    tickers = payload.get("tickers")
    if not isinstance(tickers, list) or not tickers:
        raise BadRequest("'tickers' must be a non-empty list of strings")
    cleaned = []
    for ticker in tickers:
        ticker = str(ticker).strip().upper()
        if not ticker:
            raise BadRequest("'tickers' contains an empty ticker")
        cleaned.append(ticker)
    return cleaned



def _build_sample_cov(tickers, payload) -> np.ndarray:
    """Load returns and compute sample covariance."""
    start_date = _require_date(payload, "start_date")
    end_date = _require_date(payload, "end_date")

    panel = load_prices_adjclose(
        ch_engine(), sorted(set(tickers)), start_date, end_date
    )

    missing = [t for t in tickers if t not in panel.columns]
    if missing:
        raise NotFound(
            f"no price data for {', '.join(missing)} between {start_date} and {end_date}"
        )

    returns = to_returns(panel[tickers], "simple").dropna()
    if returns.empty:
        raise NotFound("no overlapping trading days for these tickers in this range")

    return sample_covariance(returns)


def _build_factor_cov(tickers, payload) -> np.ndarray:
    """Load factor data and assemble factor-model covariance."""
    asof_date = _require_date(payload, "asof_date")
    lookback_window = int(payload.get("lookback_window", 252))
    model_version = payload.get("model_version", "bayes_ridge_v1")
    cov_method = payload.get("factor_cov_method", "ledoit_wolf")
    resid_method = payload.get("resid_method", "ols_sigma2")

    try:
        B, sigma_f, resid_var = _factor_repo.load_aligned(
            asof_date, tickers,
            lookback_window=lookback_window,
            model_version=model_version,
            cov_method=cov_method,
            resid_method=resid_method,
        )
    except LookupError as e:
        raise NotFound(str(e))

    cov = factor_model_covariance(B, sigma_f, resid_var)
    return cov, list(B.index)




@app.route("/healthz", methods=["GET"])
def healthz():
    """Health check"""
    return jsonify({"status": "ok"}), 200


@app.route("/api/optimize", methods=["POST"])
def optimize():
    """Allocate w to minimize risk subject to constraints.

    Routes to a chosen strategy (mean_variance, etc.)
    and covariance method (sample from returns, factor model, etc.).
    """
    payload = request.get_json(silent=True)
    if payload is None:
        raise BadRequest("request body must be JSON")

    strategy = _require_strategy(payload)
    cov_method = _require_cov_method(payload)
    weights_dict = _require_weights(payload)
    tickers = list(weights_dict.keys())
    w_prev = np.array([weights_dict[t] for t in tickers])

    # Build covariance matrix
    if cov_method == "sample":
        cov = _build_sample_cov(tickers, payload)
        aligned_tickers = tickers
    elif cov_method == "factor_model":
        cov, aligned_tickers = _build_factor_cov(tickers, payload)
    else:
        raise BadRequest(f"unknown cov_method: {cov_method}")

    dropped = sorted(set(tickers) - set(aligned_tickers))
    w_prev = np.array([weights_dict[t] for t in aligned_tickers] )

    # Prepare strategy kwargs
    kwargs = {
        "cov": cov,
        "w_prev": w_prev,
        "gross_target": float(payload.get("gross_target", 2.0)),
        "net_target": float(payload.get("net_target", 0.0)),
        "w_bounds": float(payload.get("w_bounds", 0.5)),
        "lambda_turnover": float(payload.get("lambda_turnover", 1.0)),
        "mu_alpha": float(payload.get("mu_alpha", 0.0)),
        "ensure_psd": True,
    }

    # Optional alpha vector
    alpha_dict = payload.get("alpha")
    if alpha_dict:
        if not isinstance(alpha_dict, dict):
            raise BadRequest("'alpha' must be a dict of {ticker: value}")
        alpha = np.array([alpha_dict.get(t, 0.0) for t in aligned_tickers])
        kwargs["alpha"] = alpha

    # Factor constraints (if provided)
    factor_limits = payload.get("factor_limits")
    if factor_limits:
        raise BadRequest("factor_limits not yet supported via HTTP (TODO)")

    # Solve
    try:
        result = solve(strategy, **kwargs)
    except ValueError as e:
        raise BadRequest(str(e))
    except RuntimeError as e:
        raise ServiceError(str(e))

    # Return
    return jsonify({
        "weights": {tickers[i]: float(result.weights[i]) for i in range(len(aligned_tickers))},
        "status": result.status,
        "risk": float(result.risk),
        "turnover": float(result.turnover),
        "gross_realized": float(result.gross_realized),
        "net_realized": float(result.net_realized),
        "solver": result.solver,
        "cov_method": cov_method,
        "droped_tickers": dropped,
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8000")))