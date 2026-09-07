"""
services/risk/app.py

VaR / ES computation and backtesting over HTTP.

This service is orchestration only. cedge_core.risk

ALPHA CONVENTION
    The two VaR methods take different risk-level parameters.

        method='parametric' -> "confidence": 0.99   (confidence level)
        method='fhs'        -> "alpha": 0.01        (tail probability)


Run locally:
    python services/risk/app.py
"""
import math

import numpy as np
from flask import Flask, jsonify, request

from cedge_core.risk.fhs_var import rolling_fhs_es, rolling_fhs_var
from cedge_core.risk.param_var import rolling_parametric_var_confidence
from cedge_core.risk.var_backtest_model import run_backtest

import pandas as pd

app = Flask(__name__)

VAR_CONVENTION = "VaR/ES are positive loss numbers; larger = riskier"

class BadRequest(Exception):
    """"""

@app.errorhandler(BadRequest)
def _handle_bad_request(exc):
    return jsonify( {'error': str(exc)}), 400

def _to_json_safe(values):
    """NaN -> None, numpy scalar -> Python float"""
    out = []
    for v in np.asarray(values, dtype=float):
        out.append(None if math.isnan(v) else float(v))
    return out



def _require_returns(payload):
    returns = payload.get("returns")
    if returns is None:
        raise BadRequest("'returns' is required")
    if not isinstance(returns, list) or len(returns) == 0:
        raise BadRequest("'returns' must be a non-empty array")
    try:
        arr = np.array(returns, dtype=float)
    except (TypeError, ValueError):
        raise BadRequest("'returns' must contain only numbers")
    if np.isnan(arr).any():
        raise BadRequest("'returns' must not contain null/NaN")

    return arr

def _require_unit_interval(payload, key, default):
    value = payload.get(key, default)
    try:
        value = float(value)
    except (TypeError, ValueError):
        raise BadRequest(f"'{key}' must be a number")
    if not 0.0 < value < 1.0:
        raise BadRequest(f"'{key}' must be strictly between 0 and 1, got {value}")
    return value

def _require_window(payload, n_obs):
    window = payload.get("window", 252)
    try:
        window = int(window)
    except (TypeError, ValueError):
        raise BadRequest("'window' must be an integer")
    if window < 2:
        raise BadRequest(f"'window' must be at least 2, got {window}")
    if window >= n_obs:
        raise BadRequest(
            f"'window' ({window}) must be smaller than the number of returns ({n_obs}); "
            "every output would be NaN otherwise"
        )
    return window

@app.route("/healthz", methods=['GET'])
def healthz():
    """Health Check"""
    return jsonify( {"status": "ok"}), 200

@app.route("/api/var_es", methods=['POST'])
def var_es():
    payload = request.get_json(silent=True)
    if payload is None:
        raise BadRequest("request body must be JSON")

    returns = _require_returns(payload)
    window = _require_window(payload, len(returns))
    method = payload.get('method', 'parametric')

    if method == 'parametric':
        confidence = _require_unit_interval(payload, "confidence", 0.95)
        vol_method = payload.get("vol_method", 'sample')
        if vol_method not in ("sample", "ewma"):
            raise BadRequest(f"'vol_method' must be 'sample' or 'ewma', got {vol_method!r}")
        frame = rolling_parametric_var_confidence (
            pd.Series(returns),
            confidence=confidence,
            window=window,
            vol_method=vol_method,
        )
        var_values, es_values = frame["var"], frame["es"]
        params = {"confidence": confidence, "window": window, "vol_method": vol_method}
    elif method == "fhs":
        alpha = _require_unit_interval(payload, "alpha", 0.01)
        lam = _require_unit_interval(payload, "lam", 0.94)
        var_values = rolling_fhs_var(returns, window=window, alpha=alpha, lam=lam)
        es_values = rolling_fhs_es(returns, window=window, alpha=alpha, lam=lam)
        params = {"alpha": alpha, "window": window, "lam": lam}

    else:
        raise BadRequest(f"'method' must be 'parametric' or 'fhs', got {method!r}")

    return jsonify({
        "method": method,
        "params": params,
        "var": _to_json_safe(var_values),
        "es": _to_json_safe(es_values),
        "convention": VAR_CONVENTION,
    })

@app.route("/api/var_backtest", methods=['POST'])
def var_backtest():
    payload = request.get_json(silent=True)
    if payload is None:
        raise BadRequest("request body must be JSON")

    returns = _require_returns(payload)
    alpha = _require_unit_interval(payload, "alpha", 0.05)

    var_est = payload.get("var_est")
    if var_est is None:
        raise BadRequest("'var_est' is required")
    var_arr = np.asarray(var_est, dtype=float)
    if var_arr.shape != returns.shape:
        raise BadRequest(
            f"'var_est' length ({var_arr.size}) must match 'returns' length ({returns.size})"
        )

    es_est = payload.get("es_est")
    es_arr = None
    if es_est is not None:
        es_arr = np.asarray(es_est, dtype=float)
        if es_arr.shape != returns.shape:
            raise BadRequest(
                f"'es_est' length ({es_arr.size}) must match 'returns' length ({returns.size})"
            )

    # The warm-up NaNs a rolling VaR carries would propagate into every test
    # statistic, so drop those days here rather than returning all-NaN results.
    valid = ~np.isnan(var_arr)
    if es_arr is not None:
        valid &= ~np.isnan(es_arr)
    if not valid.any():
        raise BadRequest("no overlapping non-null observations between 'returns' and 'var_est'")

    result = run_backtest(
        returns[valid],
        var_arr[valid],
        es_est=None if es_arr is None else es_arr[valid],
        alpha=alpha,
    )
    result["breaches"] = [int(b) for b in result["breaches"]]
    result["n_observations"] = int(valid.sum())
    return jsonify(result)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)