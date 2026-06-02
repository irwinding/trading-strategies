import numpy as np
import pandas as pd
import pytest

from momentum import (
    monthly_returns,
    formation_returns,
    risk_adjusted_return,
    cumulative_return,
)


def make_price_data(month_levels, start="2025-01-01"):
    """Daily 'Close' held constant within each calendar month at the given level.

    month_levels[0] applies to the first calendar month, [1] to the next, etc.
    Resampling to month-end therefore yields exactly `month_levels`.
    """
    rows = []
    base = pd.Period(start, freq="M")
    for i, level in enumerate(month_levels):
        p = base + i
        for d in pd.date_range(p.start_time, p.end_time, freq="D"):
            rows.append((d, float(level)))
    return pd.DataFrame(rows, columns=["Date", "Close"])


def test_monthly_returns_are_monthly_not_daily():
    # 3 month-ends at 100, 110, 121 -> exactly two monthly returns of +10%.
    # The buggy version produced ~89 daily returns instead.
    df = make_price_data([100, 110, 121])
    r = monthly_returns(df)
    assert len(r) == 2
    np.testing.assert_allclose(r.values, [0.10, 0.10], rtol=1e-9)


def test_formation_skips_most_recent_S_months_and_takes_T():
    # Returns: r1..r5; the most recent (r5) is a huge +92% spike.
    # With S=1, T=3 the spike must be skipped and exactly r2,r3,r4 returned.
    df = make_price_data([100, 101, 102, 103, 104, 200])
    f = formation_returns(df, T=3, S=1)
    assert len(f) == 3
    assert f.max() < 0.05  # the +92% most-recent spike is excluded


def test_risk_adjusted_is_mean_over_sample_std():
    # Monthly returns exactly [0.10, -0.10, 0.10].
    df = make_price_data([100, 110, 99, 108.9])
    np.testing.assert_allclose(
        risk_adjusted_return(df, T=3, S=0), 0.2886751, rtol=1e-5
    )


def test_cumulative_return_skips_recent_month():
    # P(0)=200 most recent, P(1)=110, P(2)=105, P(3)=100.
    # cumulative T=2,S=1 = P(1)/P(3) - 1 = 110/100 - 1 = 0.10 (spike excluded).
    df = make_price_data([100, 105, 110, 200])
    np.testing.assert_allclose(cumulative_return(df, T=2, S=1), 0.10, rtol=1e-9)


def test_insufficient_data_raises():
    df = make_price_data([100, 110])  # only 1 monthly return
    with pytest.raises(ValueError):
        formation_returns(df, T=12, S=1)
