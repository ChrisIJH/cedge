"""
cedge_core/portfolio/prices.py

Compatibility shim. The SQL that used to live here moved to
cedge_core.marketdata.prices (the one place that reads the prices table);
this module keeps the `get_daily_prices` port that portfolio/performance.py
and its tests are written against.

New code should take a cedge_core.marketdata.prices.PriceRepository instead.
"""
from __future__ import annotations

from typing import Optional, Protocol

import pandas as pd
from sqlalchemy.engine import Engine

from cedge_core.marketdata.prices import SqlPriceRepository as _SqlPrices


class PriceRepository(Protocol):
    def get_daily_prices(self,
                         tickers: list[str],
                         start_date: str,
                         end_date: str) -> pd.DataFrame:
        """Long form: columns [ticker, date, adj_close_price]."""
        ...


class SqlPriceRepository:
    def __init__(self, engine: Optional[Engine] = None):
        self._prices = _SqlPrices(engine)

    def get_daily_prices(self, tickers: list[str],
                         start_date: str,
                         end_date: str,
                         ) -> pd.DataFrame:
        return self._prices.get_prices(tickers, start_date, end_date, field="adj_close")


_default_repo = SqlPriceRepository()


def get_daily_prices(tickers: list[str], start_date: str, end_date: str,
                     repo: PriceRepository = _default_repo) -> pd.DataFrame:
    return repo.get_daily_prices(tickers, start_date, end_date)