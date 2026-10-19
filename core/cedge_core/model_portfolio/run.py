"""One rebalance run over many accounts. Pure: inputs in, RunResult out, no I/O.

Reconciliation: every target account lands in exactly one bucket,
n_target = n_processed + n_exception + n_skipped, or the run is FAILED_RECONCILIATION.
"""
import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

import pandas as pd

from cedge_core.model_portfolio.accounts import Account, apply_trades
from cedge_core.model_portfolio.models import ModelPortfolio
from cedge_core.model_portfolio.trades import (
    EXC_INVARIANT,
    EXC_TOO_SMALL,
    SKIPPED_IN_BAND,
    TRADED,
    Rules,
    generate_trades,
)
from cedge_core.model_portfolio.validate import EXC_INPUT, validate_inputs

OK = "OK"
ABORTED = "ABORTED"
FAILED_RECONCILIATION = "FAILED_RECONCILIATION"

PROCESSED = (TRADED,)
EXCEPTIONS = (EXC_INPUT, EXC_TOO_SMALL, EXC_INVARIANT)
SKIPPED = (SKIPPED_IN_BAND,)


@dataclass(frozen=True)
class RunInputs:
    models: Mapping[str, ModelPortfolio]
    asof: str
    rules: Rules
    code_version: str


@dataclass
class RunResult:
    run_id: str
    status: str
    abort_reason: Optional[str]
    accounts: pd.DataFrame            # one row per target account
    trades: pd.DataFrame              # one row per order
    counts: Dict[str, int] = field(default_factory=dict)
    post_accounts: List[Account] = field(default_factory=list)


def _sha(obj: object) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def make_run_id(inputs: RunInputs, accounts: Sequence[Account],
                prices: Mapping[str, float]) -> str:
    """Changes whenever anything that can change the trades changes."""
    payload = {
        "models": {k: [m.version, dict(m.weights)] for k, m in inputs.models.items()},
        "asof": inputs.asof,
        "rules": asdict(inputs.rules),
        "code_version": inputs.code_version,
        "prices": _sha({t: prices[t] for t in sorted(prices)}),
        "holdings": _sha([[a.account_id, a.model_id, a.cash, sorted(a.shares.items())]
                          for a in sorted(accounts, key=lambda a: a.account_id)]),
    }
    return _sha(payload)[:12]


def _counts(df: pd.DataFrame, n_target: int) -> Dict[str, int]:
    status = df["status"] if len(df) else pd.Series(dtype=str)
    return {
        "n_target": n_target,
        "n_processed": int(status.isin(PROCESSED).sum()),
        "n_exception": int(status.isin(EXCEPTIONS).sum()),
        "n_skipped": int(status.isin(SKIPPED).sum()),
        **{f"status_{s}": int((status == s).sum()) for s in PROCESSED + EXCEPTIONS + SKIPPED},
    }


def run(inputs: RunInputs, accounts: Sequence[Account],
        prices: Mapping[str, float]) -> RunResult:
    rid = make_run_id(inputs, accounts, prices)
    check = validate_inputs(inputs.models, accounts, prices, inputs.rules.fractional)
    if check.abort_reason:
        return RunResult(rid, ABORTED, check.abort_reason, pd.DataFrame(), pd.DataFrame(),
                         {"n_target": len(accounts)}, list(accounts))

    rows: List[Dict[str, Any]] = []
    orders: List[Dict[str, Any]] = []
    post_accounts: List[Account] = []
    for a in accounts:
        if a.account_id in check.excluded:
            rows.append({"account_id": a.account_id, "model_id": a.model_id,
                         "status": EXC_INPUT, "reason": check.excluded[a.account_id]})
            post_accounts.append(a)
            continue
        res = generate_trades(a.shares, a.cash, inputs.models[a.model_id].weights,
                              prices, inputs.rules)
        rows.append({"account_id": a.account_id, "model_id": a.model_id,
                     "status": res.status, "reason": ";".join(res.violations),
                     "flag_codes": ";".join(res.flags), "total_value": res.total_value,
                     "drift_before": res.max_drift_before, "drift_after": res.max_drift_after,
                     "turnover": res.turnover, "post_cash": res.post_cash})
        for t, q in res.trades.items():
            orders.append({"account_id": a.account_id, "ticker": t, "shares": q,
                           "side": "BUY" if q > 0 else "SELL", "price": prices[t],
                           "notional": q * prices[t]})
        traded = res.status == TRADED
        post_accounts.append(apply_trades(a, res.post_shares, res.post_cash) if traded else a)

    df = pd.DataFrame(rows)
    counts = _counts(df, len(accounts))
    balanced = counts["n_target"] == (counts["n_processed"] + counts["n_exception"]
                                      + counts["n_skipped"])
    status = OK if balanced else FAILED_RECONCILIATION
    trades = pd.DataFrame(orders, columns=["account_id", "ticker", "shares", "side",
                                           "price", "notional"])
    return RunResult(rid, status, None, df, trades, counts, post_accounts)