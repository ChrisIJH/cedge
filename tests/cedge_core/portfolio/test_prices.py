from __future__ import annotations
from pathlib import Path

import pickle
import pandas as pd
import pandas.testing as pdt

import pytest

from cedge_core.portfolio.prices import get_daily_prices, PriceRepository, SqlPriceRepository

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

    def test_default_instrument_type_excludes_spy(self, baseline):
        """
        Default instrument_type='stock' excludes SPY (stored as instrument_type='etf'
        in this DB) — this is the baseline/default behavior when no instrument_type
        is passed. build_performance() explicitly passes 'etf' for its SPY lookup
        (next task), so this default-empty result is expected only for this direct call.
        """
        result = get_daily_prices(["SPY"], "2025-01-01", "2025-06-30")
        assert result.empty
        assert baseline["raw_spy"].empty

    def test_instrument_type_etf_returns_spy(self):
        result = get_daily_prices(["SPY"], "2025-01-01", "2025-06-30", "etf")
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