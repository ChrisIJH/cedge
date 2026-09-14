"""
cedge_core/portfolio/performance.py

Orchestration layer - Imperative shell (price.py) + Functional Core (stats.py)

"""

from __future__ import annotations

import pandas as pd

from cedge_core.portfolio.prices import PriceRepository, SqlPriceRepository
from cedge_core.portfolio.stats import calc_return_stats


def _leg_return(returns: pd.DataFrame, tickers: list[str],
                weights_df: pd.DataFrame) -> pd.Series:
    """
    when weights_df['abs_weight'] use weighted return,
    otherwise, equal weight.
    """
    if 'abs_weight' in weights_df.columns:
        w = weights_df.set_index('ticker').loc[tickers, 'abs_weight']
        w = w / w.sum()
        return (returns[tickers] * w).sum(axis=1)
    return returns[tickers].mean(axis=1)


def build_performance(
    weights_df: pd.DataFrame,
    start_date: str,
    end_date: str,
    repo: PriceRepository | None = None,
) -> dict | None:
    """
    Build performance data: cumulative returns + stats.

    repo defaults to SqlPriceRepository() when omitted, so existing callers
    that call build_performance(weights_df, start, end) with no repo argument
    are unaffected. Tests can inject a fake repository.
    """
    repo = repo or SqlPriceRepository()

    tickers = weights_df['ticker'].tolist()

    prices_df = repo.get_daily_prices(tickers, start_date, end_date)

    if prices_df.empty:
        return None

    pivot = prices_df.pivot_table(
        index='date', columns='ticker', values='adj_close_price', aggfunc='last')

    returns = pivot.pct_change().dropna(how='all')


    long_tickers = weights_df[weights_df['position'] == 'LONG']['ticker'].tolist()
    short_tickers = weights_df[weights_df['position'] == 'SHORT']['ticker'].tolist()

    avail_long = [t for t in long_tickers if t in returns.columns]
    avail_short = [t for t in short_tickers if t in returns.columns]

    if not avail_long or not avail_short:
        return None

    long_return = _leg_return(returns, avail_long, weights_df)
    short_return = _leg_return(returns, avail_short, weights_df)
    portfolio_return = long_return - short_return


    cum_long = (1 + long_return).cumprod() - 1
    cum_short = (1 + short_return).cumprod() - 1
    cum_portfolio = (1 + portfolio_return).cumprod() - 1

    cum_returns = pd.DataFrame({
        'Long Leg': cum_long,
        'Short Leg': cum_short,
        'L/S Portfolio': cum_portfolio,
    })

    stats = [
        calc_return_stats(portfolio_return, 'L/S Portfolio'),
        calc_return_stats(long_return, 'Long Leg'),
        calc_return_stats(short_return, 'Short Leg'),
    ]

    spy_df = repo.get_daily_prices(['SPY'], start_date, end_date )
    if not spy_df.empty:
        spy_pivot = spy_df.pivot_table(index='date', columns='ticker', values='adj_close_price', aggfunc='last')
        spy_ret = spy_pivot['SPY'].pct_change().dropna()
        cum_returns['SPY'] = (1 + spy_ret).cumprod() - 1
        stats.append(calc_return_stats(spy_ret, 'SPY'))

    cum_returns.index = pd.to_datetime(cum_returns.index)

    return {
        "cum_returns": cum_returns,
        "stats": stats,
    }
