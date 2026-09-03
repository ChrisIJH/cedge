from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cedge_core.marketdata.weights import exposures, portfolio_returns


def _returns_frame() -> pd.DataFrame:
    idx = pd.date_range("2025-01-01", periods=3, freq="B")
    return pd.DataFrame(
        {
            "AAA": [0.10, 0.00, -0.05],
            "BBB": [0.00, 0.20, 0.10],
            "CCC": [0.01, 0.01, 0.01],
        },
        index=idx,
    )


class TestExposures:
    def test_long_only_gross_equals_net(self):
        out = exposures({"AAA": 0.6, "BBB": 0.4})
        assert out["gross"] == pytest.approx(1.0, abs=1e-12)
        assert out["net"] == pytest.approx(1.0, abs=1e-12)

    def test_long_short_gross_exceeds_net(self):
        """The distinction that makes gross worth tracking separately."""
        out = exposures({"AAA": 1.0, "BBB": -1.0})
        assert out["gross"] == pytest.approx(2.0, abs=1e-12)
        assert out["net"] == pytest.approx(0.0, abs=1e-12)

    def test_returns_plain_floats(self):
        """Values must be Python floats, not numpy scalars — these cross API and
        JSON boundaries, where numpy types are not serializable."""
        out = exposures({"AAA": 0.5, "BBB": -0.5})
        assert type(out["gross"]) is float
        assert type(out["net"]) is float


class TestPortfolioReturns:
    def test_weighted_sum_matches_hand_computation(self):
        rp, dates, exp = portfolio_returns(_returns_frame(), {"AAA": 0.6, "BBB": 0.4})
        # day 1: 0.6*0.10 + 0.4*0.00 = 0.06
        # day 2: 0.6*0.00 + 0.4*0.20 = 0.08
        # day 3: 0.6*(-0.05) + 0.4*0.10 = 0.01
        assert rp.iloc[0] == pytest.approx(0.06, abs=1e-12)
        assert rp.iloc[1] == pytest.approx(0.08, abs=1e-12)
        assert rp.iloc[2] == pytest.approx(0.01, abs=1e-12)

    def test_uses_only_the_requested_tickers(self):
        """CCC is in the frame but not in the weights — it must not contribute."""
        rp, _, _ = portfolio_returns(_returns_frame(), {"AAA": 1.0})
        assert rp.iloc[0] == pytest.approx(0.10, abs=1e-12)
        assert rp.iloc[2] == pytest.approx(-0.05, abs=1e-12)

    def test_long_short_weights(self):
        rp, _, exp = portfolio_returns(_returns_frame(), {"AAA": 1.0, "BBB": -1.0})
        # day 2: 1.0*0.00 + (-1.0)*0.20 = -0.20
        assert rp.iloc[1] == pytest.approx(-0.20, abs=1e-12)
        assert exp["gross"] == pytest.approx(2.0, abs=1e-12)
        assert exp["net"] == pytest.approx(0.0, abs=1e-12)

    def test_single_ticker_weight_one(self):
        """Degenerate case: the portfolio is just the one asset."""
        df = _returns_frame()
        rp, _, _ = portfolio_returns(df, {"BBB": 1.0})
        pd.testing.assert_series_equal(rp, df["BBB"], check_names=False)

    def test_returns_dates_alongside_series(self):
        df = _returns_frame()
        rp, dates, _ = portfolio_returns(df, {"AAA": 0.5, "BBB": 0.5})
        assert list(dates) == list(df.index)
        assert len(rp) == len(dates)

    def test_rows_with_nan_are_dropped(self):
        df = _returns_frame()
        df.loc[df.index[1], "AAA"] = np.nan
        rp, dates, _ = portfolio_returns(df, {"AAA": 0.5, "BBB": 0.5})
        assert len(rp) == 2
        assert df.index[1] not in dates

    def test_integer_weights_do_not_truncate(self):
        """Weights given as ints must not force integer arithmetic."""
        rp, _, _ = portfolio_returns(_returns_frame(), {"AAA": 1, "BBB": -1})
        assert rp.iloc[1] == pytest.approx(-0.20, abs=1e-12)


class TestPortfolioReturnsTickerCasing:
    """Keys must already be uppercase — this is the module's stated contract.

    The pre-migration version uppercased keys internally and then looked the
    uppercased key up in the original dict, so a lowercase key raised KeyError
    — the exact opposite of what the uppercasing was meant to allow. Keys are
    now used as given, so a lowercase key fails on the column lookup instead,
    which names the actual problem.
    """

    def test_uppercase_keys_work(self):
        rp, _, _ = portfolio_returns(_returns_frame(), {"AAA": 1.0})
        assert rp.iloc[0] == pytest.approx(0.10, abs=1e-12)

    def test_lowercase_key_fails_on_the_missing_column(self):
        with pytest.raises(KeyError):
            portfolio_returns(_returns_frame(), {"aaa": 1.0})
