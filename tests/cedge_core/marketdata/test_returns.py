from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from cedge_core.marketdata.returns import flag_suspect_moves, to_returns


def _prices_frame() -> pd.DataFrame:
    """Two tickers, hand-picked so the expected returns are exact in binary."""
    idx = pd.date_range("2025-01-01", periods=4, freq="B")
    return pd.DataFrame(
        {
            "AAA": [100.0, 110.0, 121.0, 132.0],   # +10%, +10%, ~+9.09%
            "BBB": [50.0, 50.0, 25.0, 50.0],       # 0%, -50%, +100%
        },
        index=idx,
    )


class TestToReturnsSimple:
    def test_first_row_dropped(self):
        out = to_returns(_prices_frame())
        assert len(out) == 3
        assert out.index[0] == pd.Timestamp("2025-01-02")

    def test_matches_hand_computed(self):
        out = to_returns(_prices_frame(), method="simple")
        assert out["AAA"].iloc[0] == pytest.approx(0.10, abs=1e-12)
        assert out["BBB"].iloc[1] == pytest.approx(-0.50, abs=1e-12)
        assert out["BBB"].iloc[2] == pytest.approx(1.00, abs=1e-12)

    def test_default_method_is_simple(self):
        pd.testing.assert_frame_equal(
            to_returns(_prices_frame()),
            to_returns(_prices_frame(), method="simple"),
        )


class TestToReturnsLog:
    def test_matches_log_of_price_ratio(self):
        px = _prices_frame()
        out = to_returns(px, method="log")
        expected = np.log(px["AAA"].iloc[1] / px["AAA"].iloc[0])
        assert out["AAA"].iloc[0] == pytest.approx(expected, abs=1e-12)

    def test_log_returns_are_additive(self):
        """The property that motivates log returns: they sum across periods."""
        px = _prices_frame()
        out = to_returns(px, method="log")
        total = np.log(px["AAA"].iloc[-1] / px["AAA"].iloc[0])
        assert out["AAA"].sum() == pytest.approx(total, abs=1e-12)

    def test_log_below_simple_for_positive_returns(self):
        """log(1+r) < r for r > 0 — a sanity check that the two differ correctly."""
        px = _prices_frame()
        simple = to_returns(px, method="simple")["AAA"].iloc[0]
        log_r = to_returns(px, method="log")["AAA"].iloc[0]
        assert log_r < simple


class TestToReturnsSeries:
    """A single price Series must work, not just a wide frame."""

    def test_series_in_series_out(self):
        s = _prices_frame()["AAA"]
        out = to_returns(s)
        assert isinstance(out, pd.Series)
        assert len(out) == 3
        assert out.iloc[0] == pytest.approx(0.10, abs=1e-12)

    def test_series_log_method(self):
        s = _prices_frame()["AAA"]
        out = to_returns(s, method="log")
        assert isinstance(out, pd.Series)
        assert out.iloc[0] == pytest.approx(np.log(110.0 / 100.0), abs=1e-12)

    def test_series_matches_same_column_of_frame(self):
        px = _prices_frame()
        from_frame = to_returns(px)["AAA"]
        from_series = to_returns(px["AAA"])
        pd.testing.assert_series_equal(from_frame, from_series)


class TestToReturnsValidation:
    def test_unknown_method_raises(self):
        with pytest.raises(ValueError, match="Unknown method"):
            to_returns(_prices_frame(), method="geometric")



class TestToReturnsDropna:
    def _ragged(self) -> pd.DataFrame:
        idx = pd.date_range("2025-01-01", periods=4, freq="B")
        return pd.DataFrame({"AAA": [100.0, 110.0, 121.0, 132.0],
                             "BBB": [np.nan, np.nan, 25.0, 50.0]}, index=idx)

    def test_any_drops_every_row_with_a_gap(self):
        out = to_returns(self._ragged())              # default dropna="any"
        assert len(out) == 1                          # only the last day has both returns

    def test_all_keeps_rows_where_some_ticker_has_a_return(self):
        out = to_returns(self._ragged(), dropna="all")
        assert len(out) == 3
        assert np.isnan(out["BBB"].iloc[0])
        assert out["AAA"].iloc[0] == pytest.approx(0.10, abs=1e-12)

    def test_unknown_dropna_raises(self):
        with pytest.raises(ValueError, match="Unknown dropna"):
            to_returns(self._ragged(), dropna="never")

class TestFlagSuspectMoves:
    def _returns(self, **cols):
        idx = pd.date_range("2026-11-02", periods=3, freq="B")
        return pd.DataFrame(cols, index=idx)

    def test_unadjusted_four_for_one_split_is_flagged(self):
        # MTUM on 2026-11-05: 272.86 -> 62.60
        r = self._returns(MTUM=[0.01, 62.60 / 272.86 - 1, 0.02])
        (warning,) = flag_suspect_moves(r)
        assert warning["ticker"] == "MTUM"
        assert warning["date"] == "2026-11-03"
        assert warning["reason"] == "price ratio close to a split factor"

    def test_unadjusted_reverse_split_is_flagged(self):
        r = self._returns(UVXY=[0.0, 3.86, 0.0])
        assert [w["date"] for w in flag_suspect_moves(r)] == ["2026-11-03"]

    def test_large_genuine_moves_are_not_flagged(self):
        # volatility ETFs: +45% and +66% days, a -33% day
        r = self._returns(UVXY=[0.45, 0.66, -0.33])
        assert flag_suspect_moves(r) == []

    def test_extreme_move_not_near_a_split_factor_is_flagged(self):
        # ratio 0.14 is more than 10% away from 1/6 and 1/8
        r = self._returns(PARA=[0.0, -0.86, 0.0])
        (warning,) = flag_suspect_moves(r)
        assert warning["reason"] == "move of 75% or more"

    def test_nan_days_are_skipped(self):
        r = self._returns(AAA=[np.nan, 0.01, -0.02])
        assert flag_suspect_moves(r) == []