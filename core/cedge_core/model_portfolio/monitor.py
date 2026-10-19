"""Band sensitivity: how many accounts each tolerance band would trade, and at what turnover."""
from dataclasses import replace
from typing import Mapping, Sequence

import pandas as pd

from cedge_core.model_portfolio.accounts import Account
from cedge_core.model_portfolio.models import ModelPortfolio
from cedge_core.model_portfolio.trades import (
    EXC_TOO_SMALL,
    TRADED,
    Rules,
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