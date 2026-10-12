"""
apps/cedge_mcp/tools.py

Tool logic, independent of the MCP framework so it can be unit-tested with a
fake HTTP session. server.py only registers these functions as MCP tools.

Rules: read-only; explicit dates (no "latest" defaults); provenance on every
result; compact outputs (daily arrays stay in the services).
"""
from __future__ import annotations

from datetime import date
from typing import Any, Dict, List, Optional

from . import provenance, summarize
from .client import CedgeClient

DEFAULT_LEVELS = [0.05, 0.01]
DEFAULT_METHODS = ["parametric", "historic"]
MAX_HISTORY_ROWS = 260


def _parse_date(name: str, value: str) -> date:
    try:
        return date.fromisoformat(value)
    except (TypeError, ValueError):
        raise ValueError(f"'{name}' must be YYYY-MM-DD, got {value!r}")


def _check_range(start: str, end: str) -> None:
    s, e = _parse_date("start", start), _parse_date("end", end)
    if s > e:
        raise ValueError(f"start ({start}) must not be after end ({end})")


def get_regime(client: CedgeClient, as_of: str) -> Dict[str, Any]:
    """Regime score, posterior, and policy computed from prices up to as_of."""
    _parse_date("as_of", as_of)
    body = client.get("regime", "/api/regime", as_of=as_of)
    return {
        "data": body,
        "provenance": provenance.build("cedge_get_regime", [
            provenance.call_record("regime", "GET", "/api/regime", {"as_of": as_of}),
        ]),
    }


def get_regime_history(client: CedgeClient, start: str, end: str,
                       score_name: str = "macro_v1") -> Dict[str, Any]:
    """Regime scores as written by the production daily job (not recomputed)."""
    _check_range(start, end)
    args = {"start": start, "end": end, "score_name": score_name}
    body = client.get("regime", "/api/regime/history", **args)
    rows = body.get("history", [])
    truncated = len(rows) > MAX_HISTORY_ROWS
    return {
        "data": {
            "score_name": score_name,
            "n_rows": len(rows),
            "truncated_to_last": MAX_HISTORY_ROWS if truncated else None,
            "history": rows[-MAX_HISTORY_ROWS:] if truncated else rows,
        },
        "provenance": provenance.build("cedge_get_regime_history", [
            provenance.call_record("regime", "GET", "/api/regime/history", args),
        ]),
    }


def list_portfolios(client: CedgeClient) -> Dict[str, Any]:
    """Every book with its keys and date coverage."""
    body = client.get("marketdata", "/api/portfolios")
    return {
        "data": body,
        "provenance": provenance.build("cedge_list_portfolios", [
            provenance.call_record("marketdata", "GET", "/api/portfolios", {}),
        ]),
    }


def var_backtest_summary(
    client: CedgeClient,
    pf_name: str,
    as_of_date: str,
    start_date: str,
    end_date: str,
    experiment_id: Optional[str] = None,
    branch: Optional[str] = None,
    source: Optional[str] = None,
    window: int = 252,
    levels: Optional[List[float]] = None,
    methods: Optional[List[str]] = None,
    n_boot: int = 1000,
) -> Dict[str, Any]:
    """Weights in force on as_of_date -> fixed-weight returns over [start_date, end_date]
    -> full backtest. Returns test statistics only."""
    _parse_date("as_of_date", as_of_date)
    _check_range(start_date, end_date)
    levels = list(levels) if levels else list(DEFAULT_LEVELS)
    methods = list(methods) if methods else list(DEFAULT_METHODS)
    calls = []

    w_args = {"pf_name": pf_name, "experiment_id": experiment_id, "branch": branch,
              "source": source, "as_of_date": as_of_date}
    snap = client.get("marketdata", "/api/portfolio_weights", **w_args)
    calls.append(provenance.call_record("marketdata", "GET", "/api/portfolio_weights", w_args))
    weights = snap["weights"]

    r_args = {"weights": weights, "start_date": start_date, "end_date": end_date}
    ret = client.post("marketdata", "/api/portfolio_returns", r_args)
    calls.append(provenance.call_record("marketdata", "POST", "/api/portfolio_returns", r_args))
    n_obs = int(ret["n_observations"])
    if window >= n_obs:
        raise ValueError(
            f"window ({window}) must be smaller than the number of returns ({n_obs}); "
            "widen the date range or shorten the window")

    b_args = {"returns": ret["returns"], "window": window, "levels": levels,
              "methods": methods, "n_boot": n_boot}
    bt = client.post("risk", "/api/full_backtest", b_args)
    calls.append(provenance.call_record("risk", "POST", "/api/full_backtest", b_args))

    return {
        "data": {
            "book": {"pf_name": pf_name, "experiment_id": experiment_id, "branch": branch,
                     "source": source, "weights_snapshot_date": snap["trade_date"],
                     "composition": summarize.summarize_weights(weights)},
            "returns": {"start_date": start_date, "end_date": end_date, "n_observations": n_obs,
                        "gross": ret.get("gross"), "net": ret.get("net"), "beta": ret.get("beta")},
            "settings": {"window": window, "levels": levels, "methods": methods, "n_boot": n_boot},
            "cells": summarize.summarize_backtest(bt["cells"]),
            "caveats": [
                ret.get("weighting", ""),
                (f"Weights are the snapshot in force on {snap['trade_date']} (requested as_of {as_of_date}); "
                 "applying them to earlier returns uses information from that later date."),
                ("Kupiec/Christoffersen/Acerbi 'reject_5pct' are 5%-level test decisions, not model verdicts; "
                 "low breach counts give low test power."),
            ],
            "convention": bt.get("convention"),
        },
        "provenance": provenance.build("cedge_var_backtest_summary", calls),
    }