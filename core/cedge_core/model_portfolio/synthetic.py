"""Synthetic accounts for the lab. Pure: a close-price panel and a seed in, account out.

Each account was last rebalanced 1-12 months ago: on that day it bought whole
shares at the close to the model (with the cash buffer). Valued at today's
close, real relative moves create the drift. Dividends are not credited to cash."""
import math
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

from cedge_core.model_portfolio.accounts import Account
from cedge_core.model_portfolio.models import ModelPortfolio


def _buy_to_model(model: ModelPortfolio, value: float, px: Mapping[str, float],
                  cash_buffer: float) -> Tuple[Dict[str, float], float]:
    shares = {t: float(math.floor(value * (1 - cash_buffer) * w / px[t]))
              for t, w in model.weights.items()}
    cash = value - sum(q * px[t] for t, q in shares.items())
    return {t: q for t, q in shares.items() if q > 0}, cash


def make_accounts(close: pd.DataFrame, models: Sequence[ModelPortfolio], asof: str,
                  n: int = 1000, seed: int = 20261014,
                  model_probs: Sequence[float] = (0.7, 0.3), cash_buffer: float = 0.02,
                  median_value: float = 150_000.0, sigma: float = 1.5,
                  min_value: float = 500.0, deposit_prob: float = 0.10) -> List[Account]:
    """close: wide close prices (index = trading dates, columns = tickers)."""
    hist = close.copy()
    hist.index = pd.to_datetime(hist.index)
    hist = hist.sort_index()
    hist = hist[hist.index <= pd.Timestamp(asof)]
    if len(hist) < 253:
        raise ValueError(f"need >= 253 trading days up to {asof}, got {len(hist)}")
    rng = np.random.default_rng(seed)
    accounts = []
    for i in range(n):
        model = models[rng.choice(len(models), p=list(model_probs))]
        value = max(min_value, median_value * math.exp(sigma * rng.standard_normal()))
        lag = int(rng.integers(21, 253))                 # 1-12 months of trading days
        day = hist.index[-1 - lag]
        shares, cash = _buy_to_model(model, value, hist.loc[day].to_dict(), cash_buffer)
        if rng.random() < deposit_prob:                  # new money since last rebalance
            cash += value * rng.uniform(0.05, 0.20)
        accounts.append(Account(f"A{i:04d}", model.model_id, shares, cash, str(day.date())))
    return accounts


def edge_accounts(models: Mapping[str, ModelPortfolio], px: Mapping[str, float],
                  legacy: str = "IEMG", cash_buffer: float = 0.02) -> List[Account]:
    """Hand-made accounts that walk every exception path; px = run-day close."""
    m = models["MOD_6040"]
    at_target, cash = _buy_to_model(m, 200_000.0, px, cash_buffer)
    ivv_only = {"IVV": float(math.floor(100_000 / px["IVV"]))}
    return [
        Account("E01_ZERO_CASH", m.model_id, ivv_only, 0.0),               # sells fund buys
        Account("E02_LEGACY", m.model_id, {**at_target, legacy: 352.0}, cash),  # full exit
        Account("E03_NEG_SHARES", m.model_id, {"IVV": -5.0}, 10_000.0),     # EXC_INPUT
        Account("E04_TINY", m.model_id, {}, 300.0),                         # EXC_TOO_SMALL
        Account("E04B_SMALL", m.model_id, {}, 1_500.0),     # trades, but no IVV: still out
        Account("E05_DEPOSIT", m.model_id, at_target, cash + 0.15 * 200_000),  # buy only
        Account("E06_NO_PRICE", m.model_id, {**at_target, "ZZZ": 10.0}, cash),  # EXC_INPUT
        Account("E07_FRACTIONAL", m.model_id, {"IVV": 10.5}, 1_000.0),      # EXC_INPUT
        Account("E08_AT_TARGET", m.model_id, at_target, cash),              # skipped
        Account("E09_UNKNOWN_MODEL", "MOD_X", at_target, cash),             # EXC_INPUT
    ]