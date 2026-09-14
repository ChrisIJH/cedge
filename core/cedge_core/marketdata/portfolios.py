"""
cedge_core/marketdata/portfolios.py

Portfolio definitions stored in the database — what books exist, and what
each held on a given date.


"""
from __future__ import annotations

from typing import Dict, Optional, Protocol, Tuple

import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

from cedge_core.db import ch_engine


class PortfolioRepository(Protocol):
    """Port - Interface that defines it can access portfolios."""
    def list_portfolio(self) -> pd.DataFrame:
        ...


    def load_weights_asof(
            self, pf_name: str, experiment_id: Optional[str], branch: Optional[str],
            source: Optional[str], asof_date: Optional[str] = None,
    ) -> Tuple[Dict[str, float], Optional[object]]:
        ...


class SqlPortfolioRepository:
    """Adapter (Imperative Shell) - Reads portfolio_weight through whatever engine given.

    `engine` defaults to the pooled MySQL singleton in production.
    """

    def __init__(self, engine: Optional[Engine] = None):
        self._engine = engine

    def _resolve_engine(self) -> Engine:
        return self._engine or ch_engine()

    def list_portfolio(self) -> pd.DataFrame:
        sql = """
            select pf_name, experiment_id, branch, source,
                   min(trade_date) as start_date, max(trade_date) as end_date,
                   count(distinct trade_date) as n_dates,
                   count(distinct ticker) as n_tickers
            from portfolio_weight
            group by pf_name, experiment_id, branch, source
            order by pf_name, experiment_id, branch, source
        """
        with self._resolve_engine().begin() as conn:
            return pd.read_sql(text(sql), conn)

    def load_weights_asof(
        self, pf_name: str, experiment_id: Optional[str] = None,
        branch: Optional[str] = None, source: Optional[str] = None,
        asof_date: Optional[str] = None,
    ) -> Tuple[Dict[str, float], Optional[object]]:
        # experiment_id/branch/source are genuinely NULL for a live book;
        # MySQL's <=> (NULL-safe equality) matches that where plain `=` would
        # match nothing.
        base = (
            " from portfolio_weight"
            " where pf_name = :pf and experiment_id <=> :exp"
            "   and branch <=> :br and source <=> :src"
        )
        params = {"pf": pf_name, "exp": experiment_id, "br": branch,
                "src": source, "asof": asof_date}

        date_sql = "select max(trade_date)" + base + (
            " and trade_date <= :asof" if asof_date else "")

        engine = self._resolve_engine()
        with engine.begin() as conn:
            trade_date = conn.execute(text(date_sql), params).scalar()
            if trade_date is None:
                return {}, None
            df = pd.read_sql(
                text("select ticker, weight" + base + " and trade_date = :td"),
                conn, params={**params, "td": trade_date},
            )

        return dict(zip(df["ticker"], df["weight"].astype(float))), trade_date