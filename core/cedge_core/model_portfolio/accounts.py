"""Account snapshot: whole-share holdings + cash, assigned to one model."""
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass(frozen=True)
class Account:
    account_id: str
    model_id: str
    shares: Dict[str, float]
    cash: float
    last_rebalanced: Optional[str] = None


def apply_trades(account: Account, post_shares: Dict[str, float], post_cash: float) -> Account:
    """Account after its trades fill at the run prices (no costs)."""
    held = {t: q for t, q in post_shares.items() if q != 0}
    return Account(account.account_id, account.model_id, held, post_cash, account.last_rebalanced)