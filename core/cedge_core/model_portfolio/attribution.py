"""Where an account's return gap to its benchmark came from: the investment
decision (model vs benchmark) or the implementation (account vs model).

Single holding period, buy-and-hold from the account's last rebalance, so
every decomposition below is exact with no multi-period linking. Pass adjusted
closes as `returns_from` so bond coupons and fund distributions count.

    account - benchmark = allocation + selection + interaction   (model vs benchmark, Brinson-Fachler)
                        + cash + rounding                        (account vs model, see implementation.py)

Brinson-Fachler, per sleeve s (model weight w_s, return r_s; benchmark W_s, B_s; benchmark total B):
    allocation  = (w_s - W_s) * (B_s - B)     overweighting sleeves that beat the benchmark
    selection   = W_s * (r_s - B_s)            holding different funds inside a sleeve
    interaction = (w_s - W_s) * (r_s - B_s)

The benchmarks below are illustrative policy portfolios with one proxy ETF per
sleeve, not published benchmarks for these models.
"""
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence

import pandas as pd

from cedge_core.model_portfolio.accounts import Account
from cedge_core.model_portfolio.implementation import implementation_effect
from cedge_core.model_portfolio.models import ModelPortfolio

SLEEVE_OF: Dict[str, str] = {
    "IVV": "US equity", 
    "IJR": "US equity",
    "EFA": "Intl equity", 
    "IEMG": "Intl equity",
    "AGG": "US bonds", 
    "TIP": "US bonds",
}

BENCHMARKS: Dict[str, ModelPortfolio] = {
    "MOD_6040": ModelPortfolio("BM_6040", "1.0", {"IVV": 0.45, "EFA": 0.15, "AGG": 0.40}),
    "MOD_8020": ModelPortfolio("BM_8020", "1.0", {"IVV": 0.60, "EFA": 0.20, "AGG": 0.20}),
}


def _sleeves(weights: Mapping[str, float], growth: Mapping[str, float],
             sleeve_of: Mapping[str, str]) -> Dict[str, Dict[str, float]]:
    """sleeve -> {"weight", "return"} for buy-and-hold holdings (growth = P_end/P_start - 1)."""
    out: Dict[str, Dict[str, float]] = {}
    for t, w in weights.items():
        s = out.setdefault(sleeve_of[t], {"weight": 0.0, "gain": 0.0})
        s["weight"] += w
        s["gain"] += w * growth[t]
    return {k: {"weight": v["weight"], "return": v["gain"] / v["weight"]}
            for k, v in out.items() if v["weight"] > 0}


def brinson(model: Mapping[str, float], benchmark: Mapping[str, float],
            growth: Mapping[str, float], sleeve_of: Mapping[str, str] = SLEEVE_OF) -> pd.DataFrame:
    """One row per sleeve. A sleeve one side does not hold takes the other side's
    return, so its selection and interaction are zero."""
    m, b = _sleeves(model, growth, sleeve_of), _sleeves(benchmark, growth, sleeve_of)
    bench_total = sum(v["weight"] * v["return"] for v in b.values())
    rows = []
    for s in sorted(set(m) | set(b)):
        w, W = m.get(s, {}).get("weight", 0.0), b.get(s, {}).get("weight", 0.0)
        B = b[s]["return"] if s in b else m[s]["return"]
        r = m[s]["return"] if s in m else B
        rows.append({"sleeve": s, "w_model": w, "w_bench": W, "r_model": r, "r_bench": B,
                     "allocation": (w - W) * (B - bench_total),
                     "selection": W * (r - B),
                     "interaction": (w - W) * (r - B)})
    return pd.DataFrame(rows)


@dataclass(frozen=True)
class Attribution:
    benchmark: float          # benchmark buy-and-hold return
    model: float              # model buy-and-hold return
    account: float
    allocation: float
    selection: float
    interaction: float
    cash: float
    rounding: float
    drift: float              # reported alongside: model buy-hold vs daily rebalanced

    @property
    def active(self) -> float:
        return self.account - self.benchmark


def attribute(account: Account, model: ModelPortfolio, benchmark: ModelPortfolio,
              close: pd.DataFrame, end: str, returns_from: Optional[pd.DataFrame] = None,
              sleeve_of: Mapping[str, str] = SLEEVE_OF) -> Attribution:
    """close values the account at the start; returns_from (adjusted closes) gives growth."""
    impl = implementation_effect(account, model.weights, close, end, returns_from)
    gx = close if returns_from is None else returns_from
    gx = gx.set_axis(pd.to_datetime(gx.index))
    g0, g1 = gx.loc[pd.Timestamp(impl.start)], gx.loc[pd.Timestamp(impl.end)]
    tickers = set(model.weights) | set(benchmark.weights)
    growth = {t: float(g1[t] / g0[t] - 1.0) for t in tickers}
    sleeves = brinson(model.weights, benchmark.weights, growth, sleeve_of)
    bench = sum(w * growth[t] for t, w in benchmark.weights.items())
    return Attribution(
        benchmark=bench, model=impl.model_buy_hold, account=impl.account,
        allocation=float(sleeves["allocation"].sum()),
        selection=float(sleeves["selection"].sum()),
        interaction=float(sleeves["interaction"].sum()),
        cash=impl.cash_effect, rounding=impl.rounding_effect, drift=impl.drift_effect,
    )


def attribution_table(accounts: Sequence[Account], models: Mapping[str, ModelPortfolio],
                      close: pd.DataFrame, end: str, returns_from: Optional[pd.DataFrame] = None,
                      benchmarks: Mapping[str, ModelPortfolio] = BENCHMARKS) -> pd.DataFrame:
    """One row per account with a start date, a known model and a benchmark for it."""
    rows: List[Dict[str, object]] = []
    for a in accounts:
        if a.last_rebalanced is None or a.model_id not in models or a.model_id not in benchmarks:
            continue
        x = attribute(a, models[a.model_id], benchmarks[a.model_id], close, end, returns_from)
        rows.append({"account_id": a.account_id, "model_id": a.model_id, "start": a.last_rebalanced,
                     **x.__dict__, "active": x.active})
    return pd.DataFrame(rows)