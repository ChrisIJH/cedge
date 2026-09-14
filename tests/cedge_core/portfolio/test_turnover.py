from __future__ import annotations

import pandas as pd
import pytest
from cedge_core.portfolio.turnover import compute_gross, compute_turnover


class TestComputeGross:
    def test_sums_absolute_weights(self):
        w = pd.Series({'AAPL': 0.3, 'MSFT': -0.2, 'GOOG': 0.1})
        assert compute_gross(w) == pytest.approx(0.6, abs=1e-9)


class TestComputeTurnover:
    def test_no_previous_weights_falls_back_to_gross(self):
        w = pd.Series({'AAPL': 0.3, 'MSFT': -0.2})
        assert compute_turnover(w, None) == pytest.approx(compute_gross(w), abs=1e-9)

    def test_empty_previous_weights_falls_back_to_gross(self):
        w = pd.Series({'AAPL': 0.3, 'MSFT': -0.2})
        assert compute_turnover(w, pd.Series(dtype=float)) == pytest.approx(compute_gross(w), abs=1e-9)

    def test_two_way_turnover_same_tickers(self):
        w = pd.Series({'AAPL': 0.3, 'MSFT': -0.2})
        w_prev = pd.Series({'AAPL': 0.1, 'MSFT': -0.1})
        # |0.3-0.1| + |-0.2-(-0.1)| = 0.2 + 0.1 = 0.3
        assert compute_turnover(w, w_prev) == pytest.approx(0.3, abs=1e-9)

    def test_two_way_turnover_ticker_added_and_dropped(self):
        w = pd.Series({'AAPL': 0.3, 'GOOG': 0.2})
        w_prev = pd.Series({'AAPL': 0.1, 'MSFT': -0.2})
        # union = {AAPL, GOOG, MSFT}
        # |0.3-0.1| + |0.2-0| + |0-(-0.2)| = 0.2 + 0.2 + 0.2 = 0.6
        assert compute_turnover(w, w_prev) == pytest.approx(0.6, abs=1e-9)