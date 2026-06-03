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
    periods_per_year: int = 12,
    annualize: bool = False,
) -> float:
    """Sharpe ratio for a sequence of periodic returns.

    Parameters
    ----------
    returns:
        Sequence of per-period returns.
    annual_rf:
        Annualised risk-free rate (e.g. 0.05 for 5%). Converted to a per-period
        equivalent internally. When None (default), the US 3-month T-bill yield
        is fetched from yfinance as of the most recent date in `returns` (if it
        is a DatetimeIndex Series) or today.
    periods_per_year:
        Number of return periods in a year (12 for monthly, 252 for daily
        trading days). Drives both the risk-free de-compounding and, when
        ``annualize`` is set, the scaling factor.
    annualize:
        When True, scale the per-period Sharpe by ``sqrt(periods_per_year)`` to
        report an annualised figure. Defaults to False (per-period Sharpe).
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

    period_rf = (1 + annual_rf) ** (1 / periods_per_year) - 1
    sharpe = (s.mean() - period_rf) / vol
    if annualize:
        sharpe *= periods_per_year ** 0.5
    return sharpe
