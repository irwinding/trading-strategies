import math
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from test_momentum import make_price_data
from common.sharpe import sharpe_ratio
from strategies import strategy1_sharpe, strategy2_sharpe, strategy3_sharpe


def rising_data(n_months=14):
    """Strictly rising prices -> cumulative return well above 50%."""
    levels = [100 * (1.05 ** i) for i in range(n_months + 1)]
    return make_price_data(levels)


def flat_data(n_months=14):
    """Flat prices -> cumulative return of 0%, below any positive threshold."""
    return make_price_data([100] * (n_months + 1))


def make_daily_data(daily_closes, start="2026-02-01"):
    """Daily price DataFrame (Date, Close) for the out-of-sample HOLDING window."""
    dates = pd.date_range(start, periods=len(daily_closes), freq="D")
    return pd.DataFrame({"Date": dates, "Close": [float(c) for c in daily_closes]})


def daily_oos_sharpe(daily_closes):
    """Expected holding-period Sharpe: daily returns, annualized, rf=0."""
    rets = pd.Series([float(c) for c in daily_closes]).pct_change().dropna().tolist()
    return sharpe_ratio(rets, annual_rf=0.0, periods_per_year=252, annualize=True)


# ---------------------------------------------------------------------------
# Strategy 1
# ---------------------------------------------------------------------------

def test_strategy1_selects_high_momentum_stocks():
    data = {"A": rising_data(), "B": flat_data()}
    holding = {"A": make_daily_data([100, 101, 102, 101, 103]), "B": make_daily_data([100, 100, 100])}
    _, selected = strategy1_sharpe(data, holding, ["A", "B"], threshold=0.5, annual_rf=0.0)
    assert selected == ["A"]


def test_strategy1_excludes_all_returns_nan():
    data = {"A": flat_data(), "B": flat_data()}
    holding = {"A": make_daily_data([100, 101]), "B": make_daily_data([100, 101])}
    sr, selected = strategy1_sharpe(data, holding, ["A", "B"], threshold=0.5, annual_rf=0.0)
    assert selected == []
    assert math.isnan(sr)


def test_strategy1_sharpe_is_computed_on_holding_window():
    # Selection happens on the (rising) formation data, but the Sharpe must come
    # from the HOLDING-window daily returns, annualized.
    data = {"A": rising_data()}
    closes = [100, 102, 101, 104, 103, 106]
    holding = {"A": make_daily_data(closes)}
    sr, selected = strategy1_sharpe(data, holding, ["A"], threshold=0.5, annual_rf=0.0)
    assert selected == ["A"]
    np.testing.assert_allclose(sr, daily_oos_sharpe(closes), rtol=1e-9)


def test_strategy1_portfolio_is_equal_weight_of_holding_returns():
    data = {"A": rising_data(), "B": rising_data()}
    a_closes = [100, 102, 101, 104]
    b_closes = [50, 50.5, 50.2, 51]
    holding = {"A": make_daily_data(a_closes), "B": make_daily_data(b_closes)}
    sr, selected = strategy1_sharpe(data, holding, ["A", "B"], threshold=0.5, annual_rf=0.0)
    assert set(selected) == {"A", "B"}

    a_ret = pd.Series(a_closes).pct_change()
    b_ret = pd.Series(b_closes).pct_change()
    port = ((a_ret + b_ret) / 2).dropna().tolist()
    expected = sharpe_ratio(port, annual_rf=0.0, periods_per_year=252, annualize=True)
    np.testing.assert_allclose(sr, expected, rtol=1e-9)


def test_strategy1_uses_tbill_when_rf_omitted():
    data = {"A": rising_data()}
    holding = {"A": make_daily_data([100, 101, 102])}
    with patch("strategies.sharpe_ratio") as mock_sr:
        mock_sr.return_value = 1.0
        strategy1_sharpe(data, holding, ["A"], threshold=0.5)
    _, call_kwargs = mock_sr.call_args
    assert call_kwargs.get("annual_rf") is None


# ---------------------------------------------------------------------------
# Strategy 2
# ---------------------------------------------------------------------------

def test_strategy2_picks_top_risk_adjusted_stock():
    data = {"A": rising_data(), "B": flat_data()}
    holding = {"A": make_daily_data([100, 101, 102]), "B": make_daily_data([100, 100, 100])}
    _, top = strategy2_sharpe(data, holding, ["A", "B"], annual_rf=0.0)
    assert top == "A"


def test_strategy2_sharpe_is_computed_on_holding_window():
    data = {"A": rising_data(), "B": flat_data()}
    closes = [100, 103, 102, 105, 104]
    holding = {"A": make_daily_data(closes), "B": make_daily_data([100, 100, 100, 100, 100])}
    sr, top = strategy2_sharpe(data, holding, ["A", "B"], annual_rf=0.0)
    assert top == "A"
    np.testing.assert_allclose(sr, daily_oos_sharpe(closes), rtol=1e-9)


def test_strategy2_uses_tbill_when_rf_omitted():
    data = {"A": rising_data()}
    holding = {"A": make_daily_data([100, 101, 102])}
    with patch("strategies.sharpe_ratio") as mock_sr:
        mock_sr.return_value = 1.0
        strategy2_sharpe(data, holding, ["A"])
    _, call_kwargs = mock_sr.call_args
    assert call_kwargs.get("annual_rf") is None


# ---------------------------------------------------------------------------
# Strategy 3  (long-short)
# ---------------------------------------------------------------------------

def test_strategy3_longs_top_shorts_bottom():
    data = {"A": rising_data(), "B": flat_data()}
    holding = {"A": make_daily_data([100, 101, 102]), "B": make_daily_data([100, 99, 98])}
    _, long_t, short_t = strategy3_sharpe(data, holding, ["A", "B"], annual_rf=0.0)
    assert long_t == "A"
    assert short_t == "B"


def test_strategy3_portfolio_return_is_long_minus_short():
    # Long leg return r_L plus short leg return (-r_S) == r_L - r_S.
    data = {"A": rising_data(), "B": flat_data()}
    a_closes = [100, 102, 101, 104]
    b_closes = [100, 99, 100, 98]
    holding = {"A": make_daily_data(a_closes), "B": make_daily_data(b_closes)}
    sr, long_t, short_t = strategy3_sharpe(data, holding, ["A", "B"], annual_rf=0.0)
    assert (long_t, short_t) == ("A", "B")

    a_ret = pd.Series(a_closes).pct_change()
    b_ret = pd.Series(b_closes).pct_change()
    port = (a_ret - b_ret).dropna().tolist()
    expected = sharpe_ratio(port, annual_rf=0.0, periods_per_year=252, annualize=True)
    np.testing.assert_allclose(sr, expected, rtol=1e-9)


def test_strategy3_uses_tbill_when_rf_omitted():
    data = {"A": rising_data(), "B": flat_data()}
    holding = {"A": make_daily_data([100, 101, 102]), "B": make_daily_data([100, 99, 98])}
    with patch("strategies.sharpe_ratio") as mock_sr:
        mock_sr.return_value = 1.0
        strategy3_sharpe(data, holding, ["A", "B"])
    _, call_kwargs = mock_sr.call_args
    assert call_kwargs.get("annual_rf") is None
