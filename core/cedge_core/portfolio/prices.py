"""
cedge_core/portfolio/prices.py

Imperative Shell - the only file in ch_api/portfolio that connects to the DB.



Previously using abstract method, but change it to duck typing.
```
from abc import ABC, abstractmethod

class PriceRepository(ABC):
    @abstractmethod
    def get_daily_prices(self, tickers: list[str], start_date: str, end_date: str,
                          instrument_type: str = 'stock') -> pd.DataFrame:
        ...


class SqlPriceRepository(PriceRepository):
    def get_daily_prices(self, tickers: list[str], 
                         start_date: str,
                         end_date: str,
                         instrument_type: str = 'stock') -> pd.DataFrame:

        engine = ch_engine()
        ...
```
Uses typing.Protocol (structural typing) rather than an ABC: a test double
just needs a matching get_daily_prices method, without importing or
inheriting from this module. The trade-off is that a missing implementation
isn't caught at instantiation time the way an ABC would catch it — it
surfaces later as an AttributeError at call time.

"""
from __future__ import annotations

import pandas as pd
from sqlalchemy import text

from cedge_core.db import ch_engine

from typing import Protocol
class PriceRepository(Protocol):
    def get_daily_prices(self, 
                         tickers: list[str], 
                         start_date: str, 
                         end_date: str) -> pd.DataFrame:
        ...



class SqlPriceRepository:
    """MySQL connection"""

    def get_daily_prices(self, tickers: list[str], 
                         start_date: str,
                         end_date: str,
                         ) -> pd.DataFrame:

        engine = ch_engine()
        q = text("""
            select p.ticker, DATE(p.price_date) as date, 
                p.adj_close_price
            from prices p
            join trading_calendar_nyse c 
                on c.cal_date = p.price_date 
            where p.ticker in :tickers
                and date(p.price_date) >= :start_date
                and date(p.price_date) <= :end_date
                and p.instrument_type NOT IN ('factor','index','mutualfund')
                and p.adj_close_price is not null
                and c.is_trading_day=1
            order by p.ticker, date
        """)

        with engine.begin() as conn:
            return pd.read_sql(q, conn, 
                               params = {'tickers': list(tickers),
                                         'start_date': start_date,
                                         'end_date': end_date,
                                         })

_default_repo = SqlPriceRepository()

def get_daily_prices(tickers: list[str], start_date: str, end_date: str,
                     instrument_type: str = 'stock',
                     repo: PriceRepository = _default_repo):
    return repo.get_daily_prices(tickers, start_date, end_date)
