"""Portfolio VaR/ES backtest engine — pure compute (no DB).
1-D 수익률 시계열에 작동 (ch_data.portfolio.portfolio_returns 결과 또는 단일 자산).
Simple returns. VaR/ES = positive loss. Look-ahead 차단: window [t-W:t] (t 제외)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from cedge_core.db import ch_engine
from cedge_core.marketdata.prices import load_prices_adjclose
from cedge_core.marketdata.returns import to_returns
from cedge_core.marketdata.weights import portfolio_returns
from cedge_core.estimators import realized_beta
from functools import reduce

from scipy import stats

from cedge_core.risk.var_backtest_model import (
    kupiec_uc, christoffersen_cc, acerbi_szekely_z2
)

def parametric_var(w, alpha=0.05):
    mu, s = np.mean(w), np.std(w, ddof=1)
    z = stats.norm.ppf(alpha)

    return -(mu + z*s)

def parametric_es(w, alpha=0.05):
    mu, s = np.mean(w), np.std(w, ddof=1)
    z = stats.norm.ppf(alpha)

    return -(mu -s*stats.norm.pdf(z)/alpha)


def historic_var(w, alpha=0.05):
    return -(np.quantile(w, alpha))


def historic_es(w, alpha=0.05):
    a = np.quantile(w, alpha)
    tail = w[w<a]
    return -tail.mean() if tail.size else -a 

_VAR = {'parametric': parametric_var, 'historic': historic_var}
_ES= {'parametric': parametric_es, 'historic': historic_es}


def rolling_var_es(returns, window, alpha, method):
    r = np.asarray(returns, float)
    n = len(r)
    idx = np.arange(window, n)
    vf, ef = _VAR[method], _ES[method]
    var = np.array([vf(r[i-window:i], alpha) for i in idx])   # r[i-W:i] = 과거, i 제외
    es  = np.array([ef(r[i-window:i], alpha) for i in idx])
    return idx, var, es


def backtest_one(returns, window=252, alpha=0.05, method="parametric", n_boot=3000):
    r = np.asarray(returns, float)
    idx, var, es = rolling_var_es(r, window, alpha, method)

    r_eval = r[idx]
    breaches = (r_eval<-var).astype(int)

    bmask = breaches.astype(bool)
    return {
        "method": method, "alpha": alpha,
        "eval_idx": idx, "r_eval": r_eval, "var": var, "es": es, "breaches": breaches,
        "kupiec": kupiec_uc(breaches, alpha),
        "christoffersen": christoffersen_cc(breaches, alpha),
        "acerbi": acerbi_szekely_z2(r_eval, var, es, alpha, n_boot=n_boot),
        "avg_pred_es":   float(es[bmask].mean())      if bmask.any() else float("nan"),
        "avg_real_loss": float(-r_eval[bmask].mean()) if bmask.any() else float("nan"),
    }

# ---- 전체 그리드 (methods × levels) ----
def full_backtest(returns, window=252, levels=(0.05, 0.01),
                  methods=("parametric", "historic"), n_boot=3000):
    return {(m, a): backtest_one(returns, window, a, m, n_boot)
            for a in levels for m in methods}



def _cell_summary(d):
    """One backtest_one cell -> flat render dict (Part 0 contract)."""
    k, c, a = d["kupiec"], d["christoffersen"], d["acerbi"]
    es_pred, es_real = d["avg_pred_es"], d["avg_real_loss"]
    return {
        "breach_rate":   float(k["breach_rate"]),
        "breach_n":      int(k["N"]),                       # low-power display
        "kupiec_reject": bool(k["reject"]),
        "christ_reject": bool(c["reject"]),
        "n11":           int(c["transitions"]["n11"]),      # consecutive breaches
        "es_real":       float(es_real),
        "es_pred":       float(es_pred),
        "es_ratio":      float(es_real / es_pred) if es_pred else float("nan"),
        "acerbi_Z2":     float(a["Z2"]),
        "acerbi_reject": bool(a["reject"]),
    }



def multi_backtest(portfolios, *, window=252, n_boot=1500,
                   price_start="1993-01-01", price_end="2026-12-31",
                   common_window=False, market="SPY", engine=None):
    """Backtest many portfolios on one shared price panel -> tidy list[dict].

    portfolios    : {name: {ticker: weight}}. Raw weights, NOT normalized — dollar-neutral
                    books have net=0 so normalize-by-net is undefined; weights used as-is
                    (matches R1 / portfolio_returns).
    common_window : True → intersect all portfolios' valid trading days and backtest each on
                    that shared window (apples-to-apples). Default False = each native period.
    market        : ticker for realized_beta regression (always loaded, even if unused by a book).

    Row per portfolio:
      name, period_start, period_end, obs, gross, net, beta, r2, alpha, beta_n,
      cells = { (method, level): _cell_summary(...) }
    Look-ahead handled inside full_backtest (rolling window). Pure compute; no UI.
    """
    eng = engine or ch_engine()

    # one price panel for the union of tickers (+ market) — single DB call.
    # NOTE: keep px RAW (do NOT to_returns the whole panel — its global dropna would
    # truncate every book to the latest-starting ticker's history).
    universe = sorted({t.upper() for w in portfolios.values() for t in w} | {market.upper()})
    px = load_prices_adjclose(eng, universe, price_start, price_end)

    # market returns on its OWN full history (for realized_beta alignment)
    mkt = (to_returns(px[[market.upper()]], "simple")[market.upper()]
           if market.upper() in px.columns else None)

    # native per-portfolio returns — compute returns from each book's OWN tickers so a
    # single late-starting ticker in the union does NOT truncate the others.
    native = {}
    for name, w in portfolios.items():
        tickers = [t.upper() for t in w]
        ret_i = to_returns(px[tickers], "simple")     # dropna over just this book's columns
        native[name] = portfolio_returns(ret_i, w)    # (rp, dates, expo)

    common_idx = None
    if common_window and native:
        common_idx = reduce(lambda a, b: a.intersection(b),
                            (dates for (_, dates, _) in native.values()))

    rows = []
    for name, w in portfolios.items():
        rp, dates, expo = native[name]
        if common_idx is not None:
            rp = rp.loc[common_idx]
            dates = rp.index
        rp_arr = np.asarray(rp, float)

        res = full_backtest(rp_arr, window=window, n_boot=n_boot)

        if mkt is not None:
            beta = realized_beta(rp_arr, mkt.reindex(dates).to_numpy(float))
        else:
            beta = {"beta": float("nan"), "alpha": float("nan"), "r2": float("nan"), "n": 0}

        rows.append({
            "name": name,
            "period_start": pd.Timestamp(min(dates)).date().isoformat(),
            "period_end":   pd.Timestamp(max(dates)).date().isoformat(),
            "obs":   int(len(rp_arr)),
            "gross": float(expo["gross"]),
            "net":   float(expo["net"]),
            "beta":  beta["beta"], "r2": beta["r2"],
            "alpha": beta["alpha"], "beta_n": beta["n"],
            "cells": {cell: _cell_summary(d) for cell, d in res.items()},
        })
    return rows