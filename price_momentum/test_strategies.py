import math
from unittest.mock import patch

import numpy as np
import pytest

from test_momentum import make_price_data
from strategies import strategy1_sharpe, strategy2_sharpe


def rising_data(n_months=14):
    """Strictly rising prices -> cumulative return well above 50%."""
    levels = [100 * (1.05 ** i) for i in range(n_months + 1)]
    return make_price_data(levels)


def flat_data(n_months=14):
    """Flat prices -> cumulative return of 0%, below any positive threshold."""
    return make_price_data([100] * (n_months + 1))


# ---------------------------------------------------------------------------
# Strategy 1
# ---------------------------------------------------------------------------

def test_strategy1_selects_high_momentum_stocks():
    data = {"A": rising_data(), "B": flat_data()}
    _, selected = strategy1_sharpe(data, ["A", "B"], threshold=0.5, annual_rf=0.0)
    assert selected == ["A"]


def test_strategy1_excludes_all_returns_nan():
    data = {"A": flat_data(), "B": flat_data()}
    sr, selected = strategy1_sharpe(data, ["A", "B"], threshold=0.5, annual_rf=0.0)
    assert selected == []
    assert math.isnan(sr)


def test_strategy1_two_stock_portfolio_is_equal_weight():
    # Two identical rising stocks -> portfolio Sharpe == single-stock Sharpe.
    data = {"A": rising_data(), "B": rising_data()}
    sr_portfolio, selected = strategy1_sharpe(data, ["A", "B"], threshold=0.5, annual_rf=0.0)
    assert set(selected) == {"A", "B"}

    from momentum import formation_returns
    from common.sharpe import sharpe_ratio
    single_sr = sharpe_ratio(formation_returns(data["A"], T=12, S=0), annual_rf=0.0)
    np.testing.assert_allclose(sr_portfolio, single_sr, rtol=1e-9)


def test_strategy1_uses_tbill_when_rf_omitted():
    data = {"A": rising_data()}
    with patch("strategies.sharpe_ratio") as mock_sr:
        mock_sr.return_value = 1.0
        strategy1_sharpe(data, ["A"], threshold=0.5)
    _, call_kwargs = mock_sr.call_args
    assert call_kwargs.get("annual_rf") is None


# ---------------------------------------------------------------------------
# Strategy 2
# ---------------------------------------------------------------------------

def test_strategy2_picks_top_risk_adjusted_stock():
    # "A" has higher, steadier returns than "B".
    data = {"A": rising_data(), "B": flat_data()}
    _, top = strategy2_sharpe(data, ["A", "B"], annual_rf=0.0)
    assert top == "A"


def test_strategy2_returns_sharpe_for_top_stock():
    data = {"A": rising_data(), "B": flat_data()}
    sr, top = strategy2_sharpe(data, ["A", "B"], annual_rf=0.0)
    assert top == "A"
    assert not math.isnan(sr)


def test_strategy2_uses_tbill_when_rf_omitted():
    data = {"A": rising_data()}
    with patch("strategies.sharpe_ratio") as mock_sr:
        mock_sr.return_value = 1.0
        strategy2_sharpe(data, ["A"])
    _, call_kwargs = mock_sr.call_args
    assert call_kwargs.get("annual_rf") is None
