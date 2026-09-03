"""
cedge_core/portfolio/stats.py
Functional Core for portfolio performance analytics. 
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

@dataclass(frozen=True)
class PerformanceStats:
    total_return: float
    annual_return: float
    annual_vol: float
    sharpe: float
    max_drawdown: float
    name: str | None = None

    def to_policy_dict(self) -> dict:
        return {
            'Total Return': self.total_return,
            'Annual Return': self.annual_return,
            'Annual Vol': self.annual_vol,
            'Sharpe': self.sharpe,
            'Max DD': self.max_drawdown,
        }

    def to_strategy_dict(self) -> dict:
        return {
            'Strategy': self.name,
            'Total Return': self.total_return,
            'Annual Return': self.annual_return,
            'Annual Vol': self.annual_vol,
            'Sharpe': self.sharpe,
            'Max Drawdown': self.max_drawdown,
        }



def compute_performance_stats(ret_series: pd.Series, 
                              name: str | None = None) -> PerformanceStats:
    """
    Annualized return/vol/Sharpe/max-drawdown for a daily return series.

    No length guard by design: on an empty series this raises ZeroDivisionError,
    matching the original calc_stats() behavior. Callers that need a zero-length
    guard (like the original calc_return_stats()) must check before calling.
    """
    annual_return = (1 + ret_series).prod() ** (252 / len(ret_series)) - 1
    annual_vol = ret_series.std() * np.sqrt(252)
    sharpe = annual_return / annual_vol if annual_vol > 0 else np.nan
    cum = (1 + ret_series).cumprod()
    total_return = cum.iloc[-1] - 1
    max_dd = (cum / cum.cummax() - 1).min()
    return PerformanceStats(total_return, annual_return, 
                            annual_vol, sharpe, max_dd, name)



def calc_stats(ret_series: pd.Series) -> dict:
    """연간화 수익률/변동성/샤프/최대낙폭/누적수익률 계산. """
    return compute_performance_stats(ret_series).to_policy_dict()


def calc_return_stats(ret_series: pd.Series, name: str) -> dict:
    """Calculate performance statistics for a return series. """
    if len(ret_series) == 0:
        return {'Strategy': name, 'Total Return': 0, 'Annual Return': 0,
                'Annual Vol': 0, 'Sharpe': 0, 'Max Drawdown': 0}
    result = compute_performance_stats(ret_series, name).to_strategy_dict()
    if pd.isna(result['Sharpe']):
        result['Sharpe'] = 0
    return result


def build_yearly_breakdown(policy_rets: dict[str, pd.Series]) -> pd.DataFrame:
    """
    policy_rets: {'A': series, 'B': series, 'SPY': series, ...}
    Returns: DataFrame [year, Policy A, Policy B, SPY, ...]
    """

    df = pd.DataFrame(policy_rets)
    years = pd.to_datetime(df.index).year
    yearly = (1+df).groupby(years).prod()-1
    yearly.index.name = 'year'
    return yearly.reset_index()


def build_cumulative(policy_rets: dict[str, pd.Series], dates: pd.Series) -> pd.DataFrame:
    """
    policy_rets: {'Policy A': series, 'Policy B': series, ...}
    Returns: DataFrame [date, Policy A, Policy B, ...]  (cumulative returns)
    """
    result = pd.DataFrame({'date': dates.values})
    for name, series in policy_rets.items():
        result[name] = (1 + series.values).cumprod() - 1
    return result