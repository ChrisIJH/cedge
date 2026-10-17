"""
cedge_core/marketdata/prices.py

The one place that reads the `prices` table.

Every rule about how a price row becomes a usable number lives here and
nowhere else:

- trading days only, joined to trading_calendar_nyse on DATE(price_date)
  (some rows are stored at 01:00:00, not midnight — an exact datetime join
  silently drops them, see PR #31)
- date filters compare DATE(price_date), for the same reason
- `field` picks the column: "adj_close" for returns and risk, "close" for
  share counts and order notionals (trades fill at the raw close)
- rows with a NULL or non-positive price are dropped
- `exclude_types` filters instrument_type; the default keeps stocks and
  ETFs. Pass () to get index-type tickers such as ^VIX
- one row per (ticker, date): the table has no unique key and real
  duplicates exist, so the last row wins

Long form out of the repository, wide form via to_wide(). Dates are
datetime.date, index name "date", tickers as given (uppercase in the DB).
"""
from __future__ import annotations

from typing import Dict, Optional, Protocol, Sequence, Tuple

import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

from cedge_core.db import ch_engine

PRICE_COLUMNS: Dict[str, str] = {"adj_close": "adj_close_price", "close": "close_price"}
DEFAULT_EXCLUDED_TYPES: Tuple[str, ...] = ("factor", "index", "mutualfund")


class PriceRepository(Protocol):
    """Port. A test double only needs this one method."""

    def get_prices(self, tickers: Sequence[str], start_date: str, end_date: str, *,
                   field: str = "adj_close",
                   exclude_types: Sequence[str] = DEFAULT_EXCLUDED_TYPES) -> pd.DataFrame:
        """Long form: columns [ticker, date, <price column>], sorted by ticker, date."""
        ...


class SqlPriceRepository:
    """MySQL adapter. `engine` defaults to the pooled singleton."""

    def __init__(self, engine: Optional[Engine] = None):
        self._engine = engine

    def _resolve_engine(self) -> Engine:
        return self._engine or ch_engine()

    def get_prices(self, tickers: Sequence[str], start_date: str, end_date: str, *,
                   field: str = "adj_close",
                   exclude_types: Sequence[str] = DEFAULT_EXCLUDED_TYPES) -> pd.DataFrame:
        if not tickers:
            raise ValueError("tickers is empty")
        column = price_column(field)
        params: Dict[str, object] = {"tickers": tuple(tickers),
                                     "start_date": start_date, "end_date": end_date}
        if exclude_types:
            params["excluded"] = tuple(exclude_types)
        sql = text(build_sql(field, exclude_types))
        with self._resolve_engine().begin() as conn:
            df = pd.read_sql(sql, conn, params=params)

        df["date"] = pd.to_datetime(df["date"]).dt.date
        df[column] = df[column].astype(float)
        return (df.drop_duplicates(subset=["ticker", "date"], keep="last")
                  .reset_index(drop=True))


def build_sql(field: str, exclude_types: Sequence[str] = DEFAULT_EXCLUDED_TYPES) -> str:
    """The one price query. Bind params: tickers, start_date, end_date[, excluded]."""
    column = price_column(field)
    type_filter = "and p.instrument_type not in :excluded" if exclude_types else ""
    return f"""
        select p.ticker, date(p.price_date) as date, p.{column} as {column}
        from prices p
        join trading_calendar_nyse c on c.cal_date = date(p.price_date)
        where p.ticker in :tickers
          and date(p.price_date) >= :start_date
          and date(p.price_date) <= :end_date
          and c.is_trading_day = 1
          and p.{column} is not null
          and p.{column} > 0
          {type_filter}
        order by p.ticker, date
    """


def price_column(field: str) -> str:
    try:
        return PRICE_COLUMNS[field]
    except KeyError:
        raise ValueError(f"unknown price field {field!r}; use one of {sorted(PRICE_COLUMNS)}")


def to_wide(long_prices: pd.DataFrame, value: Optional[str] = None) -> pd.DataFrame:
    """[ticker, date, value] -> DataFrame(index=date, columns=ticker), float.

    `value` defaults to the one column that is not ticker/date. Duplicate
    (ticker, date) rows collapse to the last one, matching the repository.
    """
    if value is None:
        others = [c for c in long_prices.columns if c not in ("ticker", "date")]
        if len(others) != 1:
            raise ValueError(f"cannot infer the value column from {list(long_prices.columns)}")
        value = others[0]
    df = long_prices.drop_duplicates(subset=["ticker", "date"], keep="last")
    wide = df.pivot(index="date", columns="ticker", values=value).sort_index()
    wide.columns.name = "ticker"
    return wide.astype(float)


# --- wide-panel conveniences used by services and risk ---------------------

def load_prices_adjclose(engine: Optional[Engine], tickers: Sequence[str],
                         start_date: str, end_date: str) -> pd.DataFrame:
    """Wide adj-close panel for returns and risk. Stocks and ETFs only."""
    repo = SqlPriceRepository(engine)
    return to_wide(repo.get_prices(tickers, start_date, end_date, field="adj_close"))


def load_prices_close(engine: Optional[Engine], tickers: Sequence[str],
                      start_date: str, end_date: str) -> pd.DataFrame:
    """Wide raw-close panel for share counts and order notionals."""
    repo = SqlPriceRepository(engine)
    return to_wide(repo.get_prices(tickers, start_date, end_date, field="close"))