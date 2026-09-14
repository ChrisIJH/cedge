from __future__ import annotations

import pickle
from pathlib import Path

import pandas.testing as pdt
import pytest
from cedge_core.portfolio.prices import (
    PriceRepository,
    SqlPriceRepository,
    get_daily_prices,
)

pytestmark = pytest.mark.db

FIXTURE = Path(__file__).parent / "fixtures" / "known_answer_baseline.pkl"


@pytest.fixture(scope='module')
def baseline() -> dict:
    with open(FIXTURE, 'rb') as f:
        return pickle.load(f)

class TestGetDailyPrices:
    def test_matches_pre_refactor_baseline_aapl_msft(self, baseline):
        result = get_daily_prices(["AAPL", "MSFT"], "2025-01-01", "2025-06-30")
        pdt.assert_frame_equal(
            result.reset_index(drop=True),
            baseline["raw_aapl_msft"].reset_index(drop=True),
        )

    def test_instrument_type_etf_returns_spy(self):
        result = get_daily_prices(["SPY"], "2025-01-01", "2025-06-30" )
        assert not result.empty
        assert set(result["ticker"].unique()) == {"SPY"}

    def test_etf_tickers_are_included(self):
        """Regression test for #10 — SPY (stored as instrument_type='etf')
        must come back from a plain call with no type override, since a
        long/short book's ticker list can mix stocks and ETFs."""
        result = get_daily_prices(["SPY"], "2025-01-01", "2025-06-30")
        assert not result.empty
        assert set(result["ticker"].unique()) == {"SPY"}

class TestSqlPriceRepository:
    def test_implements_price_repository_protocol(self):
        repo: PriceRepository = SqlPriceRepository()
        assert hasattr(repo, "get_daily_prices")

    def test_matches_module_level_function(self, baseline):
        repo = SqlPriceRepository()
        result = repo.get_daily_prices(["AAPL", "MSFT"], "2025-01-01", "2025-06-30")
        pdt.assert_frame_equal(
            result.reset_index(drop=True),
            baseline["raw_aapl_msft"].reset_index(drop=True),
        )