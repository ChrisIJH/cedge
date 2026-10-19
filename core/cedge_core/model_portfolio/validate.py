"""Input checks before trading: abort the run vs execlude one account.

Abort - a model ticker has no usable price (every account on that model is hit),
or too many accounts are excluded (a data problem, not an account problem).

Exclude - one account's own inputs are bad (EXC_INPUT with a reason)."""
import math
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence

from cedge_core.model_portfolio.accounts import Account
from cedge_core.model_portfolio.models import ModelPortfolio

EXC_INPUT = "EXC_INPUT"

PRICE_MISSING = "PRICE_MISSING"
BAD_HOLDINGS = "BAD_HOLDINGS"
UNKNOWN_MODEL = "UNKNOWN_MODEL"

MAX_EXCLUDED_FRAC = 0.05


@dataclass(frozen=True)
class Validation:
    valid: List[Account]
    excluded: Dict[str, str]
    abort_reason: Optional[str]


def _good_price(p: Optional[float]) -> bool:
    return p is not None and math.isfinite(p) and p > 0


def _good_qty(q: float, fractional: bool) -> bool:
    return math.isfinite(q) and q >= 0 and (fractional or float(q).is_integer())


def validate_inputs(models: Mapping[str, ModelPortfolio], accounts: Sequence[Account],
                    prices: Mapping[str, float], fractional: bool = False,
                    max_excluded_frac: float = MAX_EXCLUDED_FRAC) -> Validation:
    bad_model_px = sorted({t for m in models.values() for t in m.weights
                           if not _good_price(prices.get(t))})
    if bad_model_px:
        return Validation([], {}, f"ABORTED_PRICE_MISSING: {','.join(bad_model_px)}")

    valid: List[Account] = []
    excluded: Dict[str, str] = {}
    for a in accounts:
        if a.model_id not in models:
            excluded[a.account_id] = UNKNOWN_MODEL
        elif not math.isfinite(a.cash) or a.cash < 0 or \
                not all(_good_qty(q, fractional) for q in a.shares.values()):
            excluded[a.account_id] = BAD_HOLDINGS
        elif not all(_good_price(prices.get(t)) for t, q in a.shares.items() if q != 0):
            excluded[a.account_id] = PRICE_MISSING
        else:
            valid.append(a)

    if accounts and len(excluded) / len(accounts) > max_excluded_frac:
        return Validation(valid, excluded,
                          f"ABORTED_TOO_MANY_EXCLUDED: {len(excluded)}/{len(accounts)}")
    return Validation(valid, excluded, None)