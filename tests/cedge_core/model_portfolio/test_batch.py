"""Loop vs vectorized: integer orders and statuses must match exactly."""
import numpy as np
import pytest
from cedge_core.model_portfolio.batch import generate_trades_batch, to_arrays
from cedge_core.model_portfolio.models import MODELS
from cedge_core.model_portfolio.synthetic import edge_accounts, make_accounts
from cedge_core.model_portfolio.trades import Rules, generate_trades
from cedge_core.model_portfolio.validate import validate_inputs


@pytest.mark.parametrize("rules", [Rules(band=0.05, cash_buffer=0.02),
                                   Rules(band=0.01, cash_buffer=0.05, min_trade=250),
                                   Rules(band=0.02, cash_buffer=0.02, fractional=True)])
def test_loop_matches_batch(close_panel, prices, rules):
    book = make_accounts(close_panel, list(MODELS.values()), "2026-10-13") + \
        edge_accounts(MODELS, prices)
    valid = validate_inputs(MODELS, book, prices, rules.fractional).valid
    tickers, shares, cash, weights, px = to_arrays(valid, MODELS, dict(prices))
    batch = generate_trades_batch(shares, cash, weights, px, rules, tickers)

    mismatches = []
    for i, a in enumerate(valid):
        loop = generate_trades(a.shares, a.cash, MODELS[a.model_id].weights, prices, rules)
        expect = np.array([loop.trades.get(t, 0.0) for t in tickers])
        same = np.allclose(expect, batch.trades[i], rtol=0, atol=1e-12 if rules.fractional else 0)
        if not same or loop.status != batch.status[i]:
            mismatches.append((a.account_id, loop.status, batch.status[i],
                               loop.trades, dict(zip(tickers, batch.trades[i]))))
    assert mismatches == []
