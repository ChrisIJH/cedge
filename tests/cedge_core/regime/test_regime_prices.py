# tests/cedge_core/regime/test_prices.py

import pandas as pd
import pytest
from cedge_core.regime.prices import get_macro_prices


class FakeMacroRepo:
    def get_macro_prices(self, tickers, start_date, end_date):
        return pd.DataFrame({
            "ticker": ["^VIX", "UUP"],
            "date": ["2026-01-02", "2026-01-02"],
            "adj_close_price": [15.2, 27.8],
        })


def test_get_macro_prices_includes_index_type_tickers():
    df = get_macro_prices(["^VIX", "UUP"], "2026-01-01", "2026-01-31",
                           repo=FakeMacroRepo())
    assert set(df["ticker"]) == {"^VIX", "UUP"}

@pytest.mark.db
def test_get_macro_prices_keeps_vix_rows_stored_at_0100():
    # ^VIX rows from 2023-10-26 to 2025-12-11 are stored at 01:00:00; an exact
    # datetime join to the trading calendar silently dropped them.
    df = get_macro_prices(["^VIX", "UUP"], "2025-08-26", "2026-09-30")
    counts = df.groupby("ticker").size()
    assert counts["^VIX"] == counts["UUP"] == 276