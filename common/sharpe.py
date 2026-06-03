"""Sharpe ratio calculation for any return series.

    Sharpe_i = (mean(returns) - R_f^monthly) / std(returns)

where R_f^monthly is derived from an annualised risk-free rate:
    R_f^monthly = (1 + annual_rf) ** (1/12) - 1

The default risk-free rate is the US 3-month T-bill yield (^IRX via yfinance),
fetched via utils.fetch_risk_free_rate.
Returns are assumed to be at monthly frequency when converting the annual rate.
"""

from __future__ import annotations

import pandas as pd
from .utils import fetch_risk_free_rate


def sharpe_ratio(
    returns: list | pd.Series,
    annual_rf: float | None = None,
) -> float:
    """Sharpe ratio for a sequence of periodic (monthly) returns.

    Parameters
    ----------
    returns:
        Sequence of per-period returns (assumed monthly).
    annual_rf:
        Annualised risk-free rate (e.g. 0.05 for 5%). Converted to a monthly
        equivalent internally. When None (default), the US 3-month T-bill yield
        is fetched from yfinance as of the most recent date in `returns` (if it
        is a DatetimeIndex Series) or today.
    """
    s = pd.Series(returns, dtype=float)
    vol = s.std(ddof=1)
    if vol <= 0:
        return float("nan")

    if annual_rf is None:
        as_of = (
            returns.index[-1].strftime("%Y-%m-%d")
            if isinstance(returns, pd.Series) and isinstance(returns.index, pd.DatetimeIndex)
            else None
        )
        annual_rf = fetch_risk_free_rate(as_of_date=as_of)

    monthly_rf = (1 + annual_rf) ** (1 / 12) - 1
    return (s.mean() - monthly_rf) / vol
