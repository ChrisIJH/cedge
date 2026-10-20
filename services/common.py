"""
services/common.py

The HTTP conventions every cedge service follows, in one place:

- BadRequest  -> 400 {"error": ...}   the client sent something invalid
- NotFound    -> 404 {"error": ...}   well-formed request, data not available
- ServiceError-> 500 {"error": ...}   the service itself failed (solver, etc.)
- JSON bodies are required for POST; NaN never reaches the wire (-> null)
- tickers are stripped and upper-cased; dates are "YYYY-MM-DD" strings

Flask-only. Nothing here touches cedge_core or a database.
"""
from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request


class BadRequest(Exception):
    """A client-side input problem. Carries the message sent back as 400."""


class NotFound(Exception):
    """The request is well-formed but the data is not available."""


class ServiceError(Exception):
    """A service-side failure (e.g. no solver converged)."""


def register_error_handlers(app: Flask) -> None:
    @app.errorhandler(BadRequest)
    def _bad_request(exc):
        return jsonify({"error": str(exc)}), 400

    @app.errorhandler(NotFound)
    def _not_found(exc):
        return jsonify({"error": str(exc)}), 404

    @app.errorhandler(ServiceError)
    def _service_error(exc):
        return jsonify({"error": str(exc)}), 500

    @app.route("/healthz", methods=["GET"])
    def healthz():
        """Liveness check for Docker HEALTHCHECK / orchestration probes."""
        return jsonify({"status": "ok"}), 200


def json_safe(values: Iterable[Any]) -> List[Any]:
    """Sequence -> list of float | None. NaN -> None, numpy scalars -> float."""
    return [None if math.isnan(v) else float(v) for v in np.asarray(list(values), dtype=float)]


def json_safe_dict(row: Dict[str, Any]) -> Dict[str, Any]:
    """One record -> same keys, NaN floats replaced by None."""
    return {k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in row.items()}


def require_json() -> Dict[str, Any]:
    payload = request.get_json(silent=True)
    if payload is None:
        raise BadRequest("request body must be JSON")
    return payload


def require_date(payload: Dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not value:
        raise BadRequest(f"'{key}' is required (YYYY-MM-DD)")
    try:
        pd.Timestamp(value)
    except ValueError:
        raise BadRequest(f"'{key}' is not a date: {value!r}")
    return str(value)


def require_date_arg(key: str) -> str:
    """Same as require_date, for GET query parameters."""
    return require_date(request.args, key)


def _clean_ticker(raw: Any, context: str) -> str:
    ticker = str(raw).strip().upper()
    if not ticker:
        raise BadRequest(f"'{context}' contains an empty ticker")
    return ticker


def require_tickers(payload: Dict[str, Any], key: str = "tickers") -> List[str]:
    tickers = payload.get(key)
    if not isinstance(tickers, list) or not tickers:
        raise BadRequest(f"'{key}' must be a non-empty list of strings")
    return [_clean_ticker(t, key) for t in tickers]


def require_weights(payload: Dict[str, Any], key: str = "weights") -> Dict[str, float]:
    weights = payload.get(key)
    if not isinstance(weights, dict) or not weights:
        raise BadRequest(f"'{key}' must be a non-empty object of {{ticker: weight}}")
    cleaned: Dict[str, float] = {}
    for raw_ticker, raw_weight in weights.items():
        ticker = _clean_ticker(raw_ticker, key)
        try:
            weight = float(raw_weight)
        except (TypeError, ValueError):
            raise BadRequest(f"weight for {ticker!r} is not a number: {raw_weight!r}")
        if not math.isfinite(weight):
            raise BadRequest(f"weight for {ticker!r} must be finite, got {weight}")
        cleaned[ticker] = weight
    return cleaned