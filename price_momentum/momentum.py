"""Price-momentum signals implemented per the formulas in exploration.ipynb.

Time t is measured in units of 1 month, with t = 0 the most recent month.
All inputs are daily price DataFrames (a 'Date' and 'Close' column, ascending
by date) as returned by ``utils.pull_data_from_yfinance``. Returns are computed
on month-end closes, NOT on adjacent daily closes.

    R_i(t)        = P_i(t) / P_i(t+1) - 1                       (266)
    R_i^cum       = P_i(S) / P_i(S+T) - 1                       (267)
    R_i^mean      = (1/T) * sum_{t=S}^{S+T-1} R_i(t)            (268)
    R_i^risk.adj  = R_i^mean / sigma_i                          (269)
    sigma_i^2     = 1/(T-1) * sum (R_i(t) - R_i^mean)^2         (270)
"""

import pandas as pd


def monthly_close_series(price_data: pd.DataFrame) -> pd.Series:
    """Month-end closing prices, ascending by date (last value = most recent)."""
    s = price_data.set_index("Date")["Close"].sort_index()
    return s.resample("ME").last().dropna()


def monthly_returns(price_data: pd.DataFrame) -> pd.Series:
    """Month-over-month returns (Eq. 266), ascending by date."""
    return monthly_close_series(price_data).pct_change().dropna()


def formation_returns(price_data: pd.DataFrame, T: int = 12, S: int = 1) -> pd.Series:
    """The T monthly returns of the formation period, skipping the most recent
    S months (the window t = S .. S+T-1 in Eq. 268). Returned ascending."""
    r = monthly_returns(price_data)
    if len(r) < S + T:
        raise ValueError(
            f"Need at least {S + T} monthly returns for T={T}, S={S}; have {len(r)}."
        )
    end = len(r) - S          # drop the S most-recent monthly returns
    start = end - T
    return r.iloc[start:end]


def mean_monthly_return(price_data: pd.DataFrame, T: int = 12, S: int = 1) -> float:
    """Mean monthly return over the formation period (Eq. 268)."""
    return formation_returns(price_data, T, S).mean()


def monthly_volatility(price_data: pd.DataFrame, T: int = 12, S: int = 1) -> float:
    """Sample (ddof=1) monthly volatility over the formation period (Eq. 270)."""
    return formation_returns(price_data, T, S).std(ddof=1)


def risk_adjusted_return(price_data: pd.DataFrame, T: int = 12, S: int = 1) -> float:
    """Risk-adjusted mean return over the formation period (Eq. 269)."""
    f = formation_returns(price_data, T, S)
    vol = f.std(ddof=1)
    if vol <= 0:
        return float("nan")
    return f.mean() / vol


def cumulative_return(price_data: pd.DataFrame, T: int = 12, S: int = 1) -> float:
    """Cumulative return over the formation period, skipping S months (Eq. 267).

    R_cum = P(S) / P(S+T) - 1, with t = 0 the most recent month-end.
    """
    m = monthly_close_series(price_data)
    if len(m) < S + T + 1:
        raise ValueError(
            f"Need at least {S + T + 1} month-end prices for T={T}, S={S}; have {len(m)}."
        )
    p_s = m.iloc[-1 - S]          # price S months ago
    p_st = m.iloc[-1 - (S + T)]   # price S+T months ago
    return p_s / p_st - 1
