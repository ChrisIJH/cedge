"""
services/model_portfolio/app.py

Model portfolio lab over HTTP: one rebalance run over a seeded synthetic book at
real closes, the band comparison, one account's before/after, and the
account-vs-benchmark attribution.

Every request rebuilds the same book from (asof, n, seed); price panels are
cached per as-of date, so only the first request for a date reads the DB.

Run locally (from the repo root):
    PORT=8005 python -m services.model_portfolio.app
"""
import datetime as dt
import os
from functools import lru_cache
from typing import Any, Dict, List, Tuple

import pandas as pd
from cedge_core.db import ch_engine
from cedge_core.marketdata.prices import load_prices_adjclose, load_prices_close
from cedge_core.model_portfolio.accounts import Account
from cedge_core.model_portfolio.attribution import (
    BENCHMARKS,
    SLEEVE_OF,
    attribution_table,
)
from cedge_core.model_portfolio.implementation import (
    by_account_size,
    implementation_table,
)
from cedge_core.model_portfolio.models import MODELS
from cedge_core.model_portfolio.monitor import account_detail, band_table
from cedge_core.model_portfolio.run import RunInputs, run
from cedge_core.model_portfolio.synthetic import edge_accounts, make_accounts
from cedge_core.model_portfolio.trades import Rules
from cedge_core.model_portfolio.validate import validate_inputs
from flask import Flask, jsonify, request

from services.common import (
    BadRequest,
    NotFound,
    json_safe_dict,
    register_error_handlers,
    require_date_arg,
)

app = Flask(__name__)
register_error_handlers(app)

LEGACY = "IEMG"                     # held by one edge account, in no model
TICKERS = sorted({t for m in MODELS.values() for t in m.weights} | {LEGACY})
HISTORY_DAYS = 550                  # synthetic accounts rebalanced up to 252 trading days back
CODE_VERSION = os.getenv("CEDGE_CODE_VERSION", "service")
MONEY = ("total_value", "post_cash", "notional", "value_start")
BP = ("allocation", "selection", "interaction", "cash", "rounding", "drift", "active",
      "benchmark", "model", "account")


# --- inputs -------------------------------------------------------------------

def _fetch_panels(asof: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """(raw close, adjusted close), wide, HISTORY_DAYS back to asof. Reads the DB."""
    start = (dt.date.fromisoformat(asof) - dt.timedelta(days=HISTORY_DAYS)).isoformat()
    engine = ch_engine()
    return (load_prices_close(engine, TICKERS, start, asof),
            load_prices_adjclose(engine, TICKERS, start, asof))


@lru_cache(maxsize=8)
def _load_panels(asof: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    close, adj = _fetch_panels(asof)
    latest = str(close.index.max()) if not close.empty else "none"
    if latest != asof:
        raise NotFound(f"no close for {asof} (latest {latest}); not trading on stale prices")
    return close, adj


def _float_arg(key: str, default: float, lo: float, hi: float) -> float:
    try:
        value = float(request.args.get(key, default))
    except ValueError:
        raise BadRequest(f"'{key}' must be a number")
    if not lo <= value <= hi:
        raise BadRequest(f"'{key}' must be between {lo} and {hi}, got {value}")
    return value


def _int_arg(key: str, default: int, lo: int, hi: int) -> int:
    try:
        value = int(request.args.get(key, default))
    except ValueError:
        raise BadRequest(f"'{key}' must be an integer")
    if not lo <= value <= hi:
        raise BadRequest(f"'{key}' must be between {lo} and {hi}, got {value}")
    return value


def _request() -> Dict[str, Any]:
    return {
        "asof": require_date_arg("asof"),
        "band": _float_arg("band", 0.05, 0.001, 0.5),
        "cash_buffer": _float_arg("cash_buffer", 0.02, 0.0, 0.5),
        "n": _int_arg("n", 1000, 1, 20_000),
        "seed": _int_arg("seed", 20261014, 0, 2**31 - 1),
    }


def _book(q: Dict[str, Any], close: pd.DataFrame, prices: Dict[str, float],
          deposits: bool = True) -> List[Account]:
    accounts = make_accounts(close, list(MODELS.values()), q["asof"], q["n"], q["seed"],
                             deposit_prob=0.10 if deposits else 0.0)
    model_tickers = {t for m in MODELS.values() for t in m.weights}
    if deposits and model_tickers <= set(prices):
        accounts += edge_accounts(MODELS, prices, LEGACY)
    return accounts


def _prices(close: pd.DataFrame) -> Dict[str, float]:
    return {t: float(p) for t, p in close.iloc[-1].dropna().items()}


def _records(df: pd.DataFrame) -> List[Dict[str, Any]]:
    out = df.copy()
    for col in MONEY:
        if col in out.columns:
            out[col] = out[col].round(2)
    return [json_safe_dict(r) for r in out.to_dict(orient="records")]


# --- endpoints ----------------------------------------------------------------

@app.route("/api/model_portfolio/models", methods=["GET"])
def models():
    return jsonify({
        "models": {k: {"version": m.version, "weights": dict(m.weights),
                       "benchmark": {"id": BENCHMARKS[k].model_id,
                                     "weights": dict(BENCHMARKS[k].weights)}}
                   for k, m in MODELS.items()},
        "sleeves": SLEEVE_OF,
        "benchmark_note": "illustrative policy portfolios, one proxy ETF per sleeve",
    })


@app.route("/api/model_portfolio/run", methods=["GET"])
def run_book():
    """One run: summary, reconciliation, one row per account, one row per order."""
    q = _request()
    close, _ = _load_panels(q["asof"])
    prices = _prices(close)
    accounts = _book(q, close, prices)
    inputs = RunInputs(MODELS, q["asof"], Rules(band=q["band"], cash_buffer=q["cash_buffer"]),
                       CODE_VERSION)
    res = run(inputs, accounts, prices)
    if res.abort_reason:
        return jsonify({"run_id": res.run_id, "status": res.status,
                        "abort_reason": res.abort_reason, "counts": res.counts})

    c = res.counts
    rerun = run(inputs, res.post_accounts, prices)
    flags = res.accounts["flag_codes"].fillna("")
    book_value = float(res.accounts["total_value"].sum())
    gross = float(res.trades["notional"].abs().sum())
    return jsonify({
        "run_id": res.run_id, "status": res.status, "abort_reason": None,
        "asof": q["asof"], "rules": {"band": q["band"], "cash_buffer": q["cash_buffer"]},
        "counts": c,
        "reconciliation": (f"{c['n_target']} = {c['n_processed']} processed + "
                           f"{c['n_exception']} exceptions + {c['n_skipped']} skipped"),
        "flags": {f: int(flags.str.contains(f).sum()) for f in ("CASH_SHORT", "STILL_OUT_OF_BAND")},
        "n_orders": len(res.trades),
        "book_value": round(book_value, 2),
        "gross_notional": round(gross, 2),
        "book_turnover": gross / 2 / book_value if book_value else 0.0,
        "rerun_n_processed": rerun.counts.get("n_processed"),
        "accounts": _records(res.accounts),
        "trades": _records(res.trades),
    })


@app.route("/api/model_portfolio/bands", methods=["GET"])
def bands():
    q = _request()
    close, _ = _load_panels(q["asof"])
    prices = _prices(close)
    valid = validate_inputs(MODELS, _book(q, close, prices), prices).valid
    table = band_table(valid, MODELS, prices, Rules(cash_buffer=q["cash_buffer"]))
    return jsonify({"asof": q["asof"], "bands": _records(table)})


@app.route("/api/model_portfolio/account", methods=["GET"])
def account():
    """One account before and after the orders this run would send."""
    q = _request()
    account_id = request.args.get("account_id")
    if not account_id:
        raise BadRequest("'account_id' is required")
    close, _ = _load_panels(q["asof"])
    prices = _prices(close)
    book = {a.account_id: a for a in _book(q, close, prices)}
    if account_id not in book:
        raise NotFound(f"no account {account_id!r} in this book (n={q['n']}, seed={q['seed']})")
    a = book[account_id]
    check = validate_inputs(MODELS, [a], prices, max_excluded_frac=1.0)
    if account_id in check.excluded:
        return jsonify({"account_id": account_id, "model_id": a.model_id,
                        "status": "EXC_INPUT", "reason": check.excluded[account_id],
                        "holdings": a.shares, "cash": a.cash, "rows": []})
    res, table = account_detail(a, MODELS[a.model_id].weights, prices,
                                Rules(band=q["band"], cash_buffer=q["cash_buffer"]))
    return jsonify({"account_id": account_id, "model_id": a.model_id,
                    "last_rebalanced": a.last_rebalanced, "status": res.status,
                    "flags": list(res.flags), "total_value": round(res.total_value, 2),
                    "max_drift_before": res.max_drift_before,
                    "max_drift_after": res.max_drift_after,
                    "rows": _records(table)})


@app.route("/api/model_portfolio/attribution", methods=["GET"])
def attribution():
    """Account vs benchmark = allocation + selection + interaction (model vs benchmark)
    + cash + rounding (account vs model), buy-and-hold since each account's last
    rebalance. Built on a no-deposit copy of the book so new money is not counted
    as return. About 5 s for 1,000 accounts the first time; cached per (asof, n, seed)."""
    q = _request()
    return jsonify(_attribution_payload(q["asof"], q["n"], q["seed"]))


@lru_cache(maxsize=8)
def _attribution_payload(asof: str, n: int, seed: int) -> Dict[str, Any]:
    close, adj = _load_panels(asof)
    q = {"asof": asof, "n": n, "seed": seed}
    accounts = _book(q, close, _prices(close), deposits=False)
    impl = implementation_table(accounts, MODELS, close, asof, returns_from=adj)
    attr = attribution_table(accounts, MODELS, close, asof, returns_from=adj)
    per_account = attr.merge(impl[["account_id", "value_start", "cash_start"]], on="account_id")
    by_model = (attr.groupby("model_id")[list(BP)].median() * 1e4).round(1).reset_index()
    return {
        "asof": asof,
        "note": ("buy-and-hold since each account's last rebalance; values at the paid close, "
                 "returns from adjusted closes; benchmarks are illustrative"),
        "by_model_bp": _records(by_model),
        "by_size": _records(by_account_size(impl)),
        "accounts": _records(per_account[["account_id", "model_id", "start", "value_start",
                                          "cash_start", *BP]]),
    }


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
