"""Vectorized generate_trades over an (accounts x tickers) matrix.

Same rules and the same float operation order as trades.generate_trades, so the
integer orders must match the loop version exactly (tests/.../test_batch.py).
Sums run column by column on purpose: numpy's pairwise sum could reorder the
additions and flip a floor at an exact integer boundary.
"""
from dataclasses import dataclass
from typing import List, Mapping, Optional, Sequence, Tuple

import numpy as np

from cedge_core.model_portfolio.accounts import Account
from cedge_core.model_portfolio.models import ModelPortfolio
from cedge_core.model_portfolio.trades import (
    EPS,
    EXC_INVARIANT,
    EXC_TOO_SMALL,
    SKIPPED_IN_BAND,
    TRADED,
    VALUE_TOL,
    Rules,
)


@dataclass(frozen=True)
class BatchResult:
    tickers: List[str]
    trades: np.ndarray        # (n, k) shares
    status: np.ndarray        # (n,) str
    cash_short: np.ndarray    # (n,) bool
    still_out: np.ndarray     # (n,) bool


def to_arrays(accounts: Sequence[Account], models: Mapping[str, ModelPortfolio],
              prices: Mapping[str, float]
              ) -> Tuple[List[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    tickers = sorted({t for m in models.values() for t in m.weights}
                     | {t for a in accounts for t, q in a.shares.items() if q != 0})
    col = {t: j for j, t in enumerate(tickers)}
    shares = np.zeros((len(accounts), len(tickers)))
    weights = np.zeros_like(shares)
    for i, a in enumerate(accounts):
        for t, q in a.shares.items():
            shares[i, col[t]] = q
        for t, w in models[a.model_id].weights.items():
            weights[i, col[t]] = w
    cash = np.array([a.cash for a in accounts], dtype=float)
    px = np.array([prices[t] for t in tickers], dtype=float)
    return tickers, shares, cash, weights, px


def _rowsum(x: np.ndarray) -> np.ndarray:
    s = np.zeros(x.shape[0])
    for j in range(x.shape[1]):
        s = s + x[:, j]
    return s


def _top_up(trade: np.ndarray, cap: np.ndarray, held: np.ndarray, tgt: np.ndarray,
            prices: np.ndarray, total: np.ndarray, left: np.ndarray,
            rows: np.ndarray) -> np.ndarray:
    """Same as trades._top_up, one share per row per pass (first max wins ties)."""
    trade = trade.copy()
    left = left.copy()
    idx = np.arange(trade.shape[0])
    while True:
        ok = rows[:, None] & (trade < cap) & (prices <= left[:, None])
        active = ok.any(axis=1)
        if not active.any():
            return trade
        gap = np.where(ok, tgt * total[:, None] - (held + trade) * prices, -np.inf)
        best = gap.argmax(axis=1)
        r, c = idx[active], best[active]
        trade[r, c] += 1
        left[r] = left[r] - prices[c]


def generate_trades_batch(shares: np.ndarray, cash: np.ndarray, weights: np.ndarray,
                          prices: np.ndarray, rules: Optional[Rules] = None,
                          tickers: Optional[List[str]] = None) -> BatchResult:
    rules = rules or Rules()
    held = shares.astype(float)
    tgt = weights * (1 - rules.cash_buffer)

    mv = held * prices
    total = cash + _rowsum(mv)
    drift_before = np.abs(mv / total[:, None] - tgt).max(axis=1)
    need = drift_before > rules.band

    raw = (tgt * total[:, None] - mv) / prices
    if rules.fractional:
        trade = raw
    else:
        trade = np.where(raw > 0, np.floor(raw + EPS), np.ceil(raw - EPS))
    trade = np.where(tgt == 0.0, -held, trade)
    trade = np.where(np.abs(trade * prices) < rules.min_trade, 0.0, trade)
    trade = np.where(need[:, None], trade, 0.0)

    cap = trade.copy()
    notional = trade * prices
    sells = -_rowsum(np.where(trade < 0, notional, 0.0))
    buys = _rowsum(np.where(trade > 0, notional, 0.0))
    available = cash + sells - total * rules.cash_buffer
    short = need & (buys > available) & (buys > 0)
    scale = np.where(short, np.maximum(available, 0.0) / np.where(buys > 0, buys, 1.0), 1.0)
    scaled = trade * scale[:, None]
    if not rules.fractional:
        scaled = np.floor(scaled)
    trade = np.where(short[:, None] & (trade > 0), scaled, trade)
    if not rules.fractional:
        left = available - _rowsum(np.where(trade > 0, trade * prices, 0.0))
        trade = _top_up(trade, cap, held, tgt, prices, total, left, short)

    post = held + trade
    post_cash = cash - _rowsum(trade * prices)
    post_mv = post * prices
    post_total = post_cash + _rowsum(post_mv)
    drift_after = np.abs(post_mv / post_total[:, None] - tgt).max(axis=1)
    still_out = need & (drift_after > rules.band)

    bad = (post_cash < -VALUE_TOL) | (post < 0).any(axis=1) \
        | (np.abs(post_total - total) > VALUE_TOL)
    if not rules.fractional:
        bad |= (post != np.floor(post)).any(axis=1)
    has_orders = (trade != 0).any(axis=1)

    status = np.where(~need, SKIPPED_IN_BAND,
                      np.where(bad, EXC_INVARIANT,
                               np.where(has_orders, TRADED, EXC_TOO_SMALL)))
    return BatchResult(tickers or [], trade, status, short, still_out)
