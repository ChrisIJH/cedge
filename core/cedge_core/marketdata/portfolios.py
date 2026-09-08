"""
cedge_core/marketdata/portfolios.py

Portfolio definitions stored in the database — what books exist, and what
each held on a given date.


"""
from __future__ import annotations

from sqlalchemy import Engine, text
import pandas as pd
import numpy as np

from typing import Optional, Tuple, Dict



def list_portfolio(engine: Engine) -> pd.DataFrame:
    """Every distinct book in portfolio_weight, with its date coverage.

    A book is keyed by four columns, not one: the same pf_name exists under
    several experiment_ids (a live run and its what-if variants), each with
    its own branch and source. Collapsing them would merge unrelated series.

    Columns: pf_name, experiment_id, branch, source, start_date, end_date,
    n_dates, n_tickers.
    """
    sql = """
        select pf_name, experiment_id, branch, source,
               min(trade_date) as start_date, max(trade_date) as end_date,
               count(distinct trade_date) as n_dates,
               count(distinct ticker) as n_tickers
        from portfolio_weight
        group by pf_name, experiment_id, branch, source
        order by pf_name, experiment_id, branch, source
    """
    with engine.begin() as conn:
        return pd.read_sql(text(sql), conn)

    
def load_weights_asof(
        engine: Engine,
        pf_name: str,
        experiment_id: Optional[str],
        branch: Optional[str],
        source: Optional[str],
        asof_date: Optional[str] = None,
) -> Tuple[Dict[str, float], Optional[object]]:
    """The weight snapshot in force on `asof_date` — not necessarily ON it.

    Returns ({ticker: weight}, resolved_trade_date), or ({}, None) when the
    book has no rows in range.

    `experiment_id`, `branch`, and `source` are matched with <=> (MySQL's
    NULL-safe equality) because a live book stores NULL in these columns,
    and `= NULL` would silently match nothing.
    """
    base = (" from portfolio_weight "
            " where pf_name = :pf and experiment_id <=> :exp "
            "   and branch <=> :br and source <=> :src ")
    params = {"pf": pf_name, "exp": experiment_id, "br": branch,
              "src": source, "asof": asof_date}

    date_sql = "select max(trade_date)" + base + (
        " and trade_date <= :asof" if asof_date else "")

    with engine.begin() as conn:
        trade_date = conn.execute(text(date_sql), params).scalar()
        if trade_date is None:
            return {}, None
        df = pd.read_sql(
            text("select ticker, weight" + base + " and trade_date = :td"),
            conn, params={**params, "td": trade_date},
        )

    return dict(zip(df["ticker"], df["weight"].astype(float))), trade_date
