"""
cedge_core/regime/prices.py

Imperative Shell for regime — the only file in cedge_core/regime that
connects to the DB.

Separate from portfolio/prices.py's SqlPriceRepository because regime
inputs (^VIX, index-type tickers) are excluded by that repository's
`instrument_type NOT IN ('factor','index','mutualfund')` filter, which
exists correctly for stock/ETF portfolio pricing but would silently drop
regime's macro tickers. Rather than widen that filter (and risk
reintroducing factor/mutualfund noise into portfolio pricing), regime
gets its own unfiltered query.
"""
from __future__ import annotations

from typing import Protocol

import pandas as pd
from sqlalchemy import text

from cedge_core.db import ch_engine


class MacroPriceRepository(Protocol):
    def get_macro_prices(self,
                         tickers: list[str],
                         start_date: str,
                         end_date: str) -> pd.DataFrame:
        ...



class SqlMacroPriceRepository:
    """MySQL connection. No instrument_type filter — callers pass exactly
    the tickers they want (index, ETF, whatever); trading-day filtering
    still applies."""

    def get_macro_prices(self, tickers: list[str],
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
                and p.adj_close_price is not null
                and c.is_trading_day = 1
            order by p.ticker, date
        """)

        with engine.begin() as conn:
            return pd.read_sql(q, conn,
                               params={'tickers': list(tickers),
                                       'start_date': start_date,
                                       'end_date': end_date,
                                       })

_default_repo = SqlMacroPriceRepository()

def get_macro_prices(tickers: list[str], start_date: str, end_date: str,
                     repo: MacroPriceRepository = _default_repo) -> pd.DataFrame:
    return repo.get_macro_prices(tickers, start_date, end_date)