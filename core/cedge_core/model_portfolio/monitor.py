"""Monitoring views over a book: band sensitivity, and one account's before/after detail."""
from dataclasses import replace
from typing import Mapping, Sequence, Tuple

import pandas as pd

from cedge_core.model_portfolio.accounts import Account
from cedge_core.model_portfolio.models import ModelPortfolio
from cedge_core.model_portfolio.trades import (
    EXC_TOO_SMALL,
    TRADED,
    Rules,
    TradeResult,
    generate_trades,
)


def band_table(accounts: Sequence[Account], models: Mapping[str, ModelPortfolio],
               prices: Mapping[str, float], rules: Rules,
               bands: Sequence[float] = (0.01, 0.02, 0.03, 0.05)) -> pd.DataFrame:
    """Accounts are assumed validated. Triggered = TRADED + EXC_TOO_SMALL (drift > band)."""
    rows = []
    for b in bands:
        r = replace(rules, band=b)
        res = [generate_trades(a.shares, a.cash, models[a.model_id].weights, prices, r)
               for a in accounts]
        traded = [x for x in res if x.status == TRADED]
        total_value = sum(x.total_value for x in res)
        notional = sum(x.turnover * 2 * x.total_value for x in traded)
        rows.append({
            "band": b,
            "n_accounts": len(res),
            "n_triggered": sum(x.status in (TRADED, EXC_TOO_SMALL) for x in res),
            "n_traded": len(traded),
            "n_too_small": sum(x.status == EXC_TOO_SMALL for x in res),
            "n_orders": sum(len(x.trades) for x in traded),
            "turnover_book": notional / 2 / total_value if total_value else 0.0,
            "median_drift_after": (pd.Series([x.max_drift_after for x in traded]).median()
                                   if traded else 0.0),
        })
    return pd.DataFrame(rows)

def account_detail(account: Account, model: Mapping[str, float], prices: Mapping[str, float],
                   rules: Rules) -> Tuple[TradeResult, pd.DataFrame]:
    """One row per ticker plus a CASH row: shares, weights and drift before and after
    the orders this run would send. Account inputs are assumed validated."""
    res = generate_trades(account.shares, account.cash, model, prices, rules)
    total = res.total_value
    rows = []
    for t in sorted(res.post_shares):
        before = float(account.shares.get(t, 0.0))
        after = res.post_shares[t]
        target = model.get(t, 0.0) * (1 - rules.cash_buffer)
        rows.append({
            "ticker": t, "price": prices[t],
            "shares_before": before, "order": res.trades.get(t, 0.0), "shares_after": after,
            "weight_before": before * prices[t] / total, "target": target,
            "weight_after": after * prices[t] / total,
        })
    rows.append({
        "ticker": "CASH", "price": 1.0,
        "shares_before": account.cash, "order": res.post_cash - account.cash,
        "shares_after": res.post_cash,
        "weight_before": account.cash / total, "target": rules.cash_buffer,
        "weight_after": res.post_cash / total,
    })
    table = pd.DataFrame(rows)
    table["drift_before"] = table["weight_before"] - table["target"]
    table["drift_after"] = table["weight_after"] - table["target"]
    return res, table
