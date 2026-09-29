# core/cedge_core/regime/scores.py
"""
cedge_core/regime/history.py

Imperative Shell — the only file in cedge_core/regime that reads the
regime_scores table (CHResearch's precomputed daily output, not
recomputed here).
"""
from __future__ import annotations

from typing import Protocol

import pandas as pd
from sqlalchemy import text

from cedge_core.db import ch_engine


class RegimeScoreRepository(Protocol):
    def get_regime_history(self, start_date: str, end_date: str,
                           score_name: str) -> pd.DataFrame:
        ...


class SqlRegimeScoreRepository:
    def get_regime_history(self, start_date: str, end_date: str,
                           score_name: str = "macro_v1") -> pd.DataFrame:
        """
        Reads CHResearch's precomputed daily regime_scores directly,
        rather than recomputing via model.compute_regime. These are the
        values the daily cron (regime_detect02_daily_update.py) actually
        wrote and that CHResearch's optimizer/alpha_policy consumed --
        including the credit sign-invariance bug fixed in cedge's
        PR #26, not yet fixed in CHResearch's own ch_metric.py as of
        this writing. A backtest against this table reflects what was
        actually used in production, not what cedge's corrected
        score.py would produce for the same dates.
        """
        engine = ch_engine()
        q = text("""
            select as_of_date, score,
                   posterior_risk_on, posterior_soft_patch, posterior_risk_off,
                   vix_z, dxy_z, ief_z, credit_z,
                   gross_target, beta_cap
            from regime_scores
            where score_name = :score_name
              and as_of_date between :start_date and :end_date
            order by as_of_date
        """)
        with engine.begin() as conn:
            return pd.read_sql(q, conn, params={
                "score_name": score_name,
                "start_date": start_date,
                "end_date": end_date,
            })


_default_repo = SqlRegimeScoreRepository()


def get_regime_history(start_date: str, end_date: str, score_name: str = "macro_v1",
                       repo: RegimeScoreRepository = _default_repo) -> pd.DataFrame:
    return repo.get_regime_history(start_date, end_date, score_name)