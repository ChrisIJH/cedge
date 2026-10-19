"""
core/cedge_core/model_portfolio/trades.py

Pure function, no I/O. Inputs are assumed validated (validate.py): every ticker 
has a positive price, holdings are non-negative.

Rules:
- target weight of ticker i = model weight * (1 - cash_buffer), as a share of total value
- trade only if some ticker drifts more than `band` from its target.
- whole shares round toward zero: buys floor, sells ceil (never oversell, never overbuy)
- a ticker with target 0 is sold in full, in exact integer shares
- buys are funded by cash + sells above the cash buffer; if short, buys are
  scaled down proportionally and floored
- share counts stay integers; they are never re-derived from dollar amounts
- cash left over by flooring scaled buys is spent one share at a time on the most
  underweight ticker (never above its unscaled order), so a rerun has nothing left to buy
"""
import math
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Tuple

EPS = 1e-9
VALUE_TOL = 0.01

# status: exactly one per account
TRADED = "TRADED"
SKIPPED_IN_BAND = "SKIPPED_IN_BAND"
EXC_TOO_SMALL = "EXC_TOO_SMALL"
EXC_INVARIANT = "EXC_INVARIANT"

# flags: any number per account
CASH_SHORT = "CASH_SHORT"
STILL_OUT_OF_BAND = "STILL_OUT_OF_BAND"

@dataclass(frozen=True)
class Rules:
    band: float = 0.05
    cash_buffer: float = 0.02
    min_trade: float = 0.0
    fractional: bool = False

@dataclass(frozen=True)
class TradeResult:
    status: str
    flags: Tuple[str, ...]
    trades: Dict[str, float]        # ticker -> shares (+buy / -sell), non-zero only
    post_shares: Dict[str, float]
    post_cash: float
    total_value: float
    max_drift_before: float
    max_drift_after: float
    turnover: float                 # one-way: sum |trade $| / 2 / total
    violations: Tuple[str, ...] = ()

def round_toward_zero(x: float) -> int:
    if x > 0:
        return math.floor(x + EPS)
    return math.ceil(x - EPS)

def _top_up(trade: Dict[str, float], cap: Mapping[str, float], tickers: List[str],
            held: Mapping[str, float], tgt_w: Mapping[str, float],
            prices: Mapping[str, float], total: float, left: float) -> None:
    """Spend cash left by flooring, one share at a time, most underweight first."""
    while True:
        best, best_gap = None, 0.0
        for t in tickers:
            if trade[t] < cap[t] and prices[t] <= left:
                gap = tgt_w[t] * total - (held[t] + trade[t]) * prices[t]
                if best is None or gap > best_gap:
                    best, best_gap = t, gap
        if best is None:
            return
        trade[best] += 1
        left -= prices[best]


def _weights(shares: Mapping[str, float], cash: float,
             prices: Mapping[str, float]) -> Tuple[float, Dict[str, float]]:
    total = cash + sum(q * prices[t] for t, q in shares.items())
    return total, {t: q * prices[t] / total for t, q in shares.items()}


def _max_drift(cur_w: Mapping[str, float], tgt_w: Mapping[str, float]) -> float:
    return max(abs(cur_w[t] - tgt_w[t]) for t in tgt_w)


def generate_trades(shares: Mapping[str, float], cash: float,
                    model: Mapping[str, float], prices: Mapping[str, float],
                    rules: Optional[Rules] = None) -> TradeResult:
    rules = rules or Rules()
    tickers = sorted(set(model) | {t for t, q in shares.items() if q != 0})
    held = {t: float(shares.get(t, 0)) for t in tickers}
    tgt_w = {t: model.get(t, 0.0) * (1 - rules.cash_buffer) for t in tickers}

    total, cur_w = _weights(held, cash, prices)
    drift_before = _max_drift(cur_w, tgt_w)

    if drift_before <= rules.band:
        return TradeResult(SKIPPED_IN_BAND, (), {}, held, cash, total,
                           drift_before, drift_before, 0.0)

    # 1) raw trade toward full target, rounded toward zero
    trade: Dict[str, float] = {}
    for t in tickers:
        if tgt_w[t] == 0.0:
            trade[t] = -held[t]                       # full liquidation, exact
            continue
        raw = (tgt_w[t] * total - held[t] * prices[t]) / prices[t]
        trade[t] = raw if rules.fractional else round_toward_zero(raw)

    # 2) drop trades below the minimum ticket
    for t in tickers:
        if abs(trade[t] * prices[t]) < rules.min_trade:
            trade[t] = 0

    # 3) cash check: sells fund buys, cash buffer is kept if possible
    cap = dict(trade)                                 # unscaled order = upper bound
    flags = []
    sells = -sum(trade[t] * prices[t] for t in tickers if trade[t] < 0)
    buys = sum(trade[t] * prices[t] for t in tickers if trade[t] > 0)
    available = cash + sells - total * rules.cash_buffer
    if buys > available and buys > 0:
        flags.append(CASH_SHORT)
        scale = max(available, 0.0) / buys
        for t in tickers:
            if trade[t] > 0:
                scaled = trade[t] * scale
                trade[t] = scaled if rules.fractional else math.floor(scaled)
        if not rules.fractional:
            _top_up(trade, cap, tickers, held, tgt_w, prices, total,
                    available - sum(trade[t] * prices[t] for t in tickers if trade[t] > 0))

    # 4) post-trade state
    post = {t: held[t] + trade[t] for t in tickers}
    post_cash = cash - sum(trade[t] * prices[t] for t in tickers)
    post_total, post_w = _weights(post, post_cash, prices)
    drift_after = _max_drift(post_w, tgt_w)
    orders = {t: q for t, q in trade.items() if q != 0}

    if drift_after > rules.band:
        flags.append(STILL_OUT_OF_BAND)

    violations = []
    if post_cash < -VALUE_TOL:
        violations.append("negative cash")
    if any(q < 0 for q in post.values()):
        violations.append("short position")
    if abs(post_total - total) > VALUE_TOL:
        violations.append("value not conserved")
    if not rules.fractional and any(q != int(q) for q in post.values()):
        violations.append("fractional shares")

    if violations:
        status = EXC_INVARIANT
    elif orders:
        status = TRADED
    else:
        status = EXC_TOO_SMALL
    turnover = sum(abs(q * prices[t]) for t, q in orders.items()) / 2 / total
    return TradeResult(status, tuple(flags), orders, post, post_cash, total,
                       drift_before, drift_after, turnover, tuple(violations))
