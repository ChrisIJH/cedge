from __future__ import annotations
import pickle
from pathlib import Path

import pandas as pd
import pandas.testing as pdt
import pytest

from cedge_core.portfolio.performance import build_performance

FIXTURE = Path(__file__).parent / "fixtures" / "known_answer_baseline.pkl"

@pytest.fixture(scope="module")
def baseline() -> dict:
    with open(FIXTURE, "rb") as f:
        return pickle.load(f)


class FakePriceRepository:
    """Test double — returns a fixed, hand-built price panel instead of hitting the DB.

    Deliberately does NOT inherit from PriceRepository: it's a typing.Protocol,
    so matching the get_daily_prices signature is enough to satisfy it. That's
    the practical payoff of structural typing here — this test file doesn't
    import the production interface at all.

    instrument_type is accepted (to match the real interface) but ignored — this
    fake keys purely by ticker, since tests control exactly which tickers/series
    are registered.
    """

    def __init__(self, prices_by_ticker: dict[str, pd.Series]):
        self._prices_by_ticker = prices_by_ticker

    def get_daily_prices(self, tickers: list[str], start_date: str, end_date: str,
                          instrument_type: str = 'stock') -> pd.DataFrame:
        rows = []
        for ticker in tickers:
            series = self._prices_by_ticker.get(ticker)
            if series is None:
                continue
            for date, price in series.items():
                rows.append({'ticker': ticker, 'date': date, 'adj_close_price': price})
        return pd.DataFrame(rows)


class TestBuildPerformanceWithFakeRepo:
    def test_long_short_orchestration_no_db(self):
        dates = pd.date_range("2024-01-01", periods=4, freq="B")
        long_px = pd.Series([100.0, 101.0, 102.0, 103.0], index=dates)
        short_px = pd.Series([50.0, 49.5, 49.0, 48.5], index=dates)
        repo = FakePriceRepository({'LONGCO': long_px, 'SHORTCO': short_px, 'SPY': pd.Series(dtype=float)})

        weights_df = pd.DataFrame({'ticker': ['LONGCO', 'SHORTCO'], 'position': ['LONG', 'SHORT']})
        result = build_performance(weights_df, "2024-01-01", "2024-01-10", repo=repo)

        assert result is not None
        assert 'SPY' not in result['cum_returns'].columns
        assert set(result['cum_returns'].columns) == {'Long Leg', 'Short Leg', 'L/S Portfolio'}
        assert len(result['stats']) == 3
        strategy_names = {s['Strategy'] for s in result['stats']}
        assert strategy_names == {'L/S Portfolio', 'Long Leg', 'Short Leg'}

    def test_includes_spy_when_available(self):
        """
        Confirms build_performance's internal SPY lookup passes instrument_type='etf'
        end-to-end (via the fake) and wires a non-empty SPY series into the result.
        """
        dates = pd.date_range("2024-01-01", periods=4, freq="B")
        long_px = pd.Series([100.0, 101.0, 102.0, 103.0], index=dates)
        short_px = pd.Series([50.0, 49.5, 49.0, 48.5], index=dates)
        spy_px = pd.Series([400.0, 401.0, 399.0, 402.0], index=dates)
        repo = FakePriceRepository({'LONGCO': long_px, 'SHORTCO': short_px, 'SPY': spy_px})

        weights_df = pd.DataFrame({'ticker': ['LONGCO', 'SHORTCO'], 'position': ['LONG', 'SHORT']})
        result = build_performance(weights_df, "2024-01-01", "2024-01-10", repo=repo)

        assert result is not None
        assert 'SPY' in result['cum_returns'].columns
        assert len(result['stats']) == 4
        strategy_names = {s['Strategy'] for s in result['stats']}
        assert strategy_names == {'L/S Portfolio', 'Long Leg', 'Short Leg', 'SPY'}

    def test_returns_none_when_prices_empty(self):
        repo = FakePriceRepository({})
        weights_df = pd.DataFrame({'ticker': ['NOPE'], 'position': ['LONG']})
        result = build_performance(weights_df, "2024-01-01", "2024-01-10", repo=repo)
        assert result is None

@pytest.mark.db
class TestBuildPerformanceMatchesBaseline:
    def test_matches_pre_refactor_baseline(self, baseline):
        result = build_performance(baseline["weights_df"], "2025-01-01", "2025-06-30")

        if baseline["perf_before"] is None:
            assert result is None
            return

        pdt.assert_frame_equal(result["cum_returns"], baseline["perf_before"]["cum_returns"])
        assert result["stats"] == baseline["perf_before"]["stats"]


class TestWeightedLegReturn:
    def test_uses_abs_weight_when_present(self):
        """LONG1과 LONG2가 비중 0.7/0.3이면 가중평균이어야 하고, 단순평균(0.05)과는 달라야 함."""
        dates = pd.date_range("2024-01-01", periods=3, freq="B")
        long1_px = pd.Series([100.0, 110.0, 121.0], index=dates)   # 매일 +10%
        long2_px = pd.Series([100.0, 100.0, 100.0], index=dates)   # 변화 없음
        short_px = pd.Series([50.0, 49.5, 49.0], index=dates)
        repo = FakePriceRepository({'LONG1': long1_px, 'LONG2': long2_px, 'SHORTCO': short_px})

        weights_df = pd.DataFrame({
            'ticker': ['LONG1', 'LONG2', 'SHORTCO'],
            'position': ['LONG', 'LONG', 'SHORT'],
            'abs_weight': [0.7, 0.3, 1.0],
        })
        result = build_performance(weights_df, "2024-01-01", "2024-01-10", repo=repo)

        # 첫 기간(복리 없음): 0.7*0.10 + 0.3*0.0 = 0.07
        long_leg_first_period = result['cum_returns']['Long Leg'].iloc[0]
        assert long_leg_first_period == pytest.approx(0.07, abs=1e-9)

    def test_falls_back_to_equal_weight_without_abs_weight_column(self):
        """abs_weight 컬럼이 없으면 기존과 동일한 단순평균."""
        dates = pd.date_range("2024-01-01", periods=3, freq="B")
        long1_px = pd.Series([100.0, 110.0, 121.0], index=dates)
        long2_px = pd.Series([100.0, 100.0, 100.0], index=dates)
        short_px = pd.Series([50.0, 49.5, 49.0], index=dates)
        repo = FakePriceRepository({'LONG1': long1_px, 'LONG2': long2_px, 'SHORTCO': short_px})

        weights_df = pd.DataFrame({
            'ticker': ['LONG1', 'LONG2', 'SHORTCO'],
            'position': ['LONG', 'LONG', 'SHORT'],
        })
        result = build_performance(weights_df, "2024-01-01", "2024-01-10", repo=repo)

        # 첫 기간, 단순평균: (0.10 + 0.0) / 2 = 0.05
        long_leg_first_period = result['cum_returns']['Long Leg'].iloc[0]
        assert long_leg_first_period == pytest.approx(0.05, abs=1e-9)