"""Implementation effect: how much of an account's return gap to its model 
came from how the model was applied. not from the model itself.

One holding period, start = the account's last rebalance, end = as_of. The 
account is assumed to have held its shares and cash unchanged in between
(not brades, no deposits or withdrawals; cash earns nothing.)

Prices: `close` values the account at the start (what it actually paid);
`returns_from` (default: `close`) supplies growth. Pass adjusted closes there so
fund distributions count — with raw closes a bond ETF's coupon disappears from
its return.

    model_rebalanced    model weights held exactly every day (daily constant mix)
    model_buy_hold      model weights bought exactly at start, then left alone
    account             the accouant's whole shares + cash, left alone

    drift_effect        = model_buy_hold - model_rebalanced (not rebalancing in between)
    cash_effect         = -c0 * model_buy_hold              (c0 = cash weight at start)
    rounding_effect     = (1 - c0) * (invested - model_buy_hold)
    total               = account - model_rebalanced = drift + cash + rounding (exact)

`invested` is the return of the account's securities alone. rounding_effect is
everything that makes them differ from the model: whole-share rounding and any
off-model position still held.
"""
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from cedge_core.model_portfolio.accounts import Account
from cedge_core.model_portfolio.models import ModelPortfolio


@dataclass(frozen=True)
class ImplementationEffect:
    start: str
    end: str
    value_start: float
    cash_start: float
    model_rebalanced: float
    model_buy_hold: float
    account: float
    drift_effect: float
    cash_effect: float
    rounding_effect: float
    total: float



def implementation_effect(account: Account, model: Mapping[str, float],
                          close: pd.DataFrame, end: str,
                          returns_from: Optional[pd.DataFrame] = None) -> ImplementationEffect:
    """close / returns_from: wide prices (index = dates, columns = tickers),
    covering account.last_rebalanced .. end."""
    if account.last_rebalanced is None:
        raise ValueError(f"{account.account_id}: no last_rebalanced date")
    px = close.set_axis(pd.to_datetime(close.index)).sort_index()
    gx = px if returns_from is None else \
        returns_from.set_axis(pd.to_datetime(returns_from.index)).sort_index()
    start_ts, end_ts = pd.Timestamp(account.last_rebalanced), pd.Timestamp(end)
    for frame in (px, gx):
        if start_ts not in frame.index or end_ts not in frame.index or start_ts >= end_ts:
            raise ValueError(f"{account.account_id}: need prices on {account.last_rebalanced} and {end}")
    growth = gx.loc[end_ts] / gx.loc[start_ts]          # 1 + period return per ticker

    w = pd.Series(dict(model), dtype=float)
    daily = gx.loc[start_ts:end_ts, w.index].pct_change().iloc[1:]
    model_rebalanced = float(np.prod(1.0 + daily.to_numpy() @ w.to_numpy()) - 1.0)
    model_buy_hold = float((w * growth[w.index]).sum() - 1.0)

    q = pd.Series(account.shares, dtype=float)
    value_held = q * px.loc[start_ts, q.index]
    held0 = float(value_held.sum())
    held1 = float((value_held * growth[q.index]).sum())
    value0 = held0 + account.cash
    acct = (held1 + account.cash) / value0 - 1.0
    c0 = account.cash / value0
    invested = held1 / held0 - 1.0 if held0 > 0 else 0.0

    drift = model_buy_hold - model_rebalanced
    cash = -c0 * model_buy_hold
    rounding = (1.0 - c0) * (invested - model_buy_hold)
    return ImplementationEffect(
        start=str(start_ts.date()), end=str(end_ts.date()), value_start=value0, cash_start=c0,
        model_rebalanced=model_rebalanced, model_buy_hold=model_buy_hold, account=acct,
        drift_effect=drift, cash_effect=cash, rounding_effect=rounding,
        total=acct - model_rebalanced,
    )


def implementation_table(accounts: Sequence[Account], models: Mapping[str, ModelPortfolio],
                         close: pd.DataFrame, end: str,
                         returns_from: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """One row per account that has a last_rebalanced date and a known model."""
    rows: List[Dict[str, object]] = []
    for a in accounts:
        if a.last_rebalanced is None or a.model_id not in models:
            continue
        e = implementation_effect(a, models[a.model_id].weights, close, end, returns_from)
        rows.append({"account_id": a.account_id, "model_id": a.model_id, **e.__dict__})
    return pd.DataFrame(rows)


SIZE_EDGES = (0.0, 25_000.0, 100_000.0, 500_000.0, float("inf"))
SIZE_LABELS = ("<$25k", "$25k-100k", "$100k-500k", ">$500k")


def by_account_size(table: pd.DataFrame) -> pd.DataFrame:
    """Median effects (basis points) and worst total by starting account size."""
    t = table.assign(size=pd.cut(table["value_start"], SIZE_EDGES, labels=SIZE_LABELS))
    bp = 1e4
    return t.groupby("size", observed=True).agg(
        accounts=("account_id", "size"),
        cash_start=("cash_start", "median"),
        drift_bp=("drift_effect", lambda s: s.median() * bp),
        cash_bp=("cash_effect", lambda s: s.median() * bp),
        rounding_bp=("rounding_effect", lambda s: s.median() * bp),
        total_bp=("total", lambda s: s.median() * bp),
        worst_total_bp=("total", lambda s: s.min() * bp),
    ).reset_index()


