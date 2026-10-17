"""
cedge_core/regime/prices.py

Compatibility shim. Regime's macro inputs include index-type tickers
(^VIX), which the default price query excludes; this used to justify a
second SQL query. It is now one parameter — `exclude_types=()` — on the
single loader in cedge_core.marketdata.prices.

Kept so services/regime and the regime tests keep their `get_macro_prices`
port with a fake repository.
"""
from __future__ import annotations

from typing import Optional, Protocol

import pandas as pd
from sqlalchemy.engine import Engine

from cedge_core.marketdata.prices import SqlPriceRepository as _SqlPrices


class MacroPriceRepository(Protocol):
    def get_macro_prices(self,
                         tickers: list[str],
                         start_date: str,
                         end_date: str) -> pd.DataFrame:
        """Long form: columns [ticker, date, adj_close_price]."""
        ...


class SqlMacroPriceRepository:
    """No instrument_type filter — callers pass exactly the tickers they want."""

    def __init__(self, engine: Optional[Engine] = None):
        self._prices = _SqlPrices(engine)

    def get_macro_prices(self, tickers: list[str],
                         start_date: str,
                         end_date: str,
                         ) -> pd.DataFrame:
        return self._prices.get_prices(tickers, start_date, end_date,
                                       field="adj_close", exclude_types=())


_default_repo = SqlMacroPriceRepository()


def get_macro_prices(tickers: list[str], start_date: str, end_date: str,
                     repo: MacroPriceRepository = _default_repo) -> pd.DataFrame:
    return repo.get_macro_prices(tickers, start_date, end_date)