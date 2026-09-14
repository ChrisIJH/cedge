from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from cedge_core.marketdata.returns import to_returns


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
