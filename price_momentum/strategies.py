"""Sharpe ratio calculations for price-momentum strategies.

Tickers are *selected* on the formation period (``data``) but every Sharpe ratio
is computed **out-of-sample** on the realised daily returns of the subsequent
holding period (``holding_data``) — the same window the PnL plots use. Daily
returns are annualised (252 trading days).

Strategy 1 – long an equal-weight portfolio of stocks whose T-month cumulative
return exceeds a threshold (default >= 50%, no skip).

Strategy 2 – long the single stock with the highest risk-adjusted return over
the T-month formation period (T=12, S=1 skip).

Strategy 3 – long the top and short the bottom risk-adjusted name (dollar-neutral
long-short); the per-period return is r_long - r_short.
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd

from momentum import cumulative_return, formation_returns, risk_adjusted_return
from common.sharpe import sharpe_ratio

TRADING_DAYS_PER_YEAR = 252


def _daily_returns(price_data: pd.DataFrame) -> pd.Series:
    """Daily returns over the holding window, ascending by date."""
    s = price_data.set_index("Date")["Close"].sort_index()
    return s.pct_change().dropna()


def strategy1_sharpe(
    data: dict,
    holding_data: dict,
    tickers: list[str],
    threshold: float = 0.5,
    T: int = 12,
    S: int = 0,
    annual_rf: float | None = None,
) -> tuple[float, list[str]]:
    """Out-of-sample Sharpe for an equal-weight portfolio of high-momentum stocks.

    Selects tickers whose cumulative return over the T-month formation period
    (no skip, S=0) is >= threshold, then computes the annualised Sharpe ratio on
    the equal-weight portfolio's daily holding-period returns.

    Returns
    -------
    (sharpe, selected_tickers)
    """
    selected = [t for t in tickers if cumulative_return(data[t], T=T, S=S) >= threshold]
    if not selected:
        return float("nan"), []

    returns_df = pd.concat(
        [_daily_returns(holding_data[t]).rename(t) for t in selected], axis=1
    )
    portfolio_returns = returns_df.mean(axis=1)
    sharpe = sharpe_ratio(
        portfolio_returns,
        annual_rf=annual_rf,
        periods_per_year=TRADING_DAYS_PER_YEAR,
        annualize=True,
    )
    return sharpe, selected


def strategy2_sharpe(
    data: dict,
    holding_data: dict,
    tickers: list[str],
    T: int = 12,
    S: int = 1,
    annual_rf: float | None = None,
) -> tuple[float, str]:
    """Out-of-sample Sharpe for the top risk-adjusted momentum stock.

    Picks the single ticker with the highest risk-adjusted return over the
    T-month formation period (skipping the most recent S months) and computes
    the annualised Sharpe ratio on its daily holding-period returns.

    Returns
    -------
    (sharpe, top_ticker)
    """
    rar = {t: risk_adjusted_return(data[t], T=T, S=S) for t in tickers}
    top_ticker = max(rar, key=rar.get)
    returns = _daily_returns(holding_data[top_ticker])
    sharpe = sharpe_ratio(
        returns,
        annual_rf=annual_rf,
        periods_per_year=TRADING_DAYS_PER_YEAR,
        annualize=True,
    )
    return sharpe, top_ticker


def strategy3_sharpe(
    data: dict,
    holding_data: dict,
    tickers: list[str],
    T: int = 12,
    S: int = 1,
    annual_rf: float | None = None,
) -> tuple[float, str, str]:
    """Out-of-sample Sharpe for a long-short risk-adjusted momentum portfolio.

    Long the highest and short the lowest risk-adjusted name over the T-month
    formation period (skipping the most recent S months). The dollar-neutral
    portfolio's per-period return is r_long - r_short (the short leg profits when
    it falls), and the annualised Sharpe is computed on the daily holding-period
    returns.

    Returns
    -------
    (sharpe, long_ticker, short_ticker)
    """
    rar = {t: risk_adjusted_return(data[t], T=T, S=S) for t in tickers}
    sorted_tickers = sorted(rar, key=rar.get, reverse=True)
    long_ticker = sorted_tickers[0]
    short_ticker = sorted_tickers[-1]

    long_returns = _daily_returns(holding_data[long_ticker])
    short_returns = _daily_returns(holding_data[short_ticker])
    portfolio_returns = (long_returns - short_returns).dropna()
    sharpe = sharpe_ratio(
        portfolio_returns,
        annual_rf=annual_rf,
        periods_per_year=TRADING_DAYS_PER_YEAR,
        annualize=True,
    )
    return sharpe, long_ticker, short_ticker
