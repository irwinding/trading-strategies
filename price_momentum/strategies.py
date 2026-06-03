"""Sharpe ratio calculations for price-momentum strategies.

Strategy 1 – long stocks whose 12-month cumulative return exceeds a
threshold (default ≥ 50%).  The portfolio is equally weighted across all
selected stocks.

Strategy 2 – long the single stock with the highest risk-adjusted return
over the 12-month formation period (T=12, S=1 skip).
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd

from momentum import cumulative_return, formation_returns, risk_adjusted_return
from common.sharpe import sharpe_ratio


def strategy1_sharpe(
    data: dict,
    tickers: list[str],
    threshold: float = 0.5,
    T: int = 12,
    S: int = 0,
    annual_rf: float | None = None,
) -> tuple[float, list[str]]:
    """Sharpe ratio for an equal-weight portfolio of high-momentum stocks.

    Selects tickers whose cumulative return over the T-month formation
    period (no skip, S=0) is >= threshold, then computes the Sharpe ratio
    on the equal-weight portfolio's monthly returns.

    Returns
    -------
    (sharpe, selected_tickers)
    """
    selected = [t for t in tickers if cumulative_return(data[t], T=T, S=S) >= threshold]
    if not selected:
        return float("nan"), []

    returns_df = pd.concat(
        [formation_returns(data[t], T=T, S=S).rename(t) for t in selected], axis=1
    )
    portfolio_returns = returns_df.mean(axis=1)
    return sharpe_ratio(portfolio_returns, annual_rf=annual_rf), selected


def strategy2_sharpe(
    data: dict,
    tickers: list[str],
    T: int = 12,
    S: int = 1,
    annual_rf: float | None = None,
) -> tuple[float, str]:
    """Sharpe ratio for the top risk-adjusted momentum stock.

    Picks the single ticker with the highest risk-adjusted return over the
    T-month formation period (skipping the most recent S months) and
    computes its Sharpe ratio.

    Returns
    -------
    (sharpe, top_ticker)
    """
    rar = {t: risk_adjusted_return(data[t], T=T, S=S) for t in tickers}
    top_ticker = max(rar, key=rar.get)
    returns = formation_returns(data[top_ticker], T=T, S=S)
    return sharpe_ratio(returns, annual_rf=annual_rf), top_ticker
