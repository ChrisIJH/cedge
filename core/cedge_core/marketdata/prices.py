"""
cedge_core/marketdata/prices.py

"""
from __future__ import annotations

from typing import List

import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine


def load_prices_adjclose(
        engine: Engine,
        tickers: List[str],
        start_date: str,
        end_date: str
) -> pd.DataFrame:
    """Wide adj-close prices (index=price_date[date], columns=ticker).
    NYSE 거래일만; factor/index/mutualfund 제외; adj_close > 0."""

    sql = """
    select date(p.price_date) as price_date, p.ticker, p.adj_close_price
    from prices p
    join trading_calendar_nyse c on c.cal_date = date(p.price_date)
    where c.is_trading_day = 1
      and p.ticker IN :tickers
      and p.price_date >= :start_date
      and p.price_date <= :end_date
      and p.adj_close_price IS NOT NULL
      and p.instrument_type NOT IN ('factor','index','mutualfund')
      and p.adj_close_price > 0
    order by p.price_date asc
    """
    params = {"tickers": tuple(tickers), "start_date": start_date, "end_date": end_date}
    
    with engine.begin() as conn:
        df = pd.read_sql(text(sql), conn, params=params)

    df["price_date"] = pd.to_datetime(df['price_date']).dt.date
    df = df.drop_duplicates(subset=["price_date", "ticker"])

    return df.pivot(index="price_date", 
                    columns=['ticker'], 
                    values="adj_close_price").sort_index().astype(float)











