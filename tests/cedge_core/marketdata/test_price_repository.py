"""cedge_core.marketdata.prices — the single price loader.

No-DB tests cover the query text and the long->wide conversion; the `db`
tests pin the behaviours the three old loaders used to disagree on.
"""
import datetime as dt

import pandas as pd
import pytest
from cedge_core.marketdata.prices import (
    DEFAULT_EXCLUDED_TYPES,
    SqlPriceRepository,
    build_sql,
    load_prices_adjclose,
    load_prices_close,
    price_column,
    to_wide,
)


class TestBuildSql:
    def test_joins_and_filters_on_calendar_date_not_datetime(self):
        sql = build_sql("adj_close")
        assert "c.cal_date = date(p.price_date)" in sql
        assert "date(p.price_date) >= :start_date" in sql
        assert "p.price_date >= :start_date" not in sql       # the PR #31 class of bug

    def test_close_field_selects_close_column(self):
        assert "p.close_price as close_price" in build_sql("close")
        assert "adj_close" not in build_sql("close")

    def test_type_filter_is_optional(self):
        assert "instrument_type not in :excluded" in build_sql("adj_close")
        assert "instrument_type" not in build_sql("adj_close", exclude_types=())
        assert DEFAULT_EXCLUDED_TYPES == ("factor", "index", "mutualfund")

    def test_unknown_field_rejected(self):
        with pytest.raises(ValueError, match="unknown price field"):
            price_column("open")


class TestToWide:
    def _long(self):
        return pd.DataFrame({
            "ticker": ["AAA", "AAA", "BBB", "BBB", "BBB"],
            "date": [dt.date(2026, 1, 2), dt.date(2026, 1, 5),
                     dt.date(2026, 1, 2), dt.date(2026, 1, 5), dt.date(2026, 1, 5)],
            "adj_close_price": [10.0, 11.0, 20.0, 21.0, 99.0],   # BBB 01-05 duplicated
        })

    def test_shape_and_values(self):
        wide = to_wide(self._long())
        assert list(wide.columns) == ["AAA", "BBB"]
        assert list(wide.index) == [dt.date(2026, 1, 2), dt.date(2026, 1, 5)]
        assert wide.loc[dt.date(2026, 1, 2), "AAA"] == 10.0

    def test_duplicate_row_last_wins(self):
        wide = to_wide(self._long())
        assert wide.loc[dt.date(2026, 1, 5), "BBB"] == 99.0

    def test_value_column_inferred_or_explicit(self):
        long = self._long()
        pd.testing.assert_frame_equal(to_wide(long), to_wide(long, value="adj_close_price"))
        with pytest.raises(ValueError, match="infer"):
            to_wide(long.assign(extra=1.0))


@pytest.mark.db
class TestAgainstDatabase:
    TICKERS = ("AAPL", "MSFT", "SPY", "IVV")

    def test_long_form_contract(self):
        df = SqlPriceRepository().get_prices(list(self.TICKERS), "2025-01-01", "2025-06-30")
        assert list(df.columns) == ["ticker", "date", "adj_close_price"]
        assert df.dtypes["adj_close_price"] == float
        assert isinstance(df["date"].iloc[0], dt.date)
        assert not df.duplicated(["ticker", "date"]).any()
        assert df.equals(df.sort_values(["ticker", "date"]).reset_index(drop=True))

    def test_index_tickers_need_the_filter_off(self):
        repo = SqlPriceRepository()
        with_filter = repo.get_prices(["^VIX", "UUP"], "2025-08-26", "2026-09-30")
        assert set(with_filter["ticker"]) == {"UUP"}
        no_filter = repo.get_prices(["^VIX", "UUP"], "2025-08-26", "2026-09-30", exclude_types=())
        counts = no_filter.groupby("ticker").size()
        assert counts["^VIX"] == counts["UUP"] == 276          # 01:00:00 rows kept

    def test_close_and_adj_close_share_one_shape(self):
        adj = load_prices_adjclose(None, ["IVV", "AGG"], "2026-09-01", "2026-10-13")
        raw = load_prices_close(None, ["IVV", "AGG"], "2026-09-01", "2026-10-13")
        assert adj.shape == raw.shape
        assert list(adj.columns) == list(raw.columns) == ["AGG", "IVV"]
        assert (raw > 0).all().all()