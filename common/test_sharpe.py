import math
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from common.sharpe import sharpe_ratio


def test_explicit_zero_rf():
    # returns [0.10, -0.10, 0.10]; mean=0.0333, std=0.1155 -> 0.2887
    returns = [0.10, -0.10, 0.10]
    np.testing.assert_allclose(sharpe_ratio(returns, annual_rf=0.0), 0.2886751, rtol=1e-5)


def test_nonzero_rf_reduces_sharpe():
    returns = [0.10, -0.10, 0.10]
    assert sharpe_ratio(returns, annual_rf=0.05) < sharpe_ratio(returns, annual_rf=0.0)


def test_rf_conversion_is_monthly_compounding():
    returns = [0.10, -0.10, 0.10]
    monthly_rf = 1.05 ** (1 / 12) - 1
    mean, std = np.mean(returns), np.std(returns, ddof=1)
    np.testing.assert_allclose(
        sharpe_ratio(returns, annual_rf=0.05), (mean - monthly_rf) / std, rtol=1e-9
    )


def test_default_rf_fetches_tbill():
    returns = [0.10, -0.10, 0.10]
    with patch("common.sharpe.fetch_risk_free_rate", return_value=0.05) as mock_fetch:
        result = sharpe_ratio(returns)
    mock_fetch.assert_called_once_with(as_of_date=None)
    monthly_rf = 1.05 ** (1 / 12) - 1
    mean, std = np.mean(returns), np.std(returns, ddof=1)
    np.testing.assert_allclose(result, (mean - monthly_rf) / std, rtol=1e-9)


def test_default_rf_uses_series_end_date():
    # A DatetimeIndex Series: the last date should be passed to fetch_risk_free_rate.
    idx = pd.date_range("2024-01-31", periods=3, freq="ME")
    returns = pd.Series([0.10, -0.10, 0.10], index=idx)
    with patch("common.sharpe.fetch_risk_free_rate", return_value=0.04) as mock_fetch:
        sharpe_ratio(returns)
    mock_fetch.assert_called_once_with(as_of_date="2024-03-31")


def test_zero_volatility_returns_nan():
    # Use exact integer values so std is truly 0 (0.05 is not representable in float)
    assert math.isnan(sharpe_ratio([0, 0, 0], annual_rf=0.0))


def test_accepts_pandas_series():
    s = pd.Series([0.10, -0.10, 0.10])
    np.testing.assert_allclose(sharpe_ratio(s, annual_rf=0.0), 0.2886751, rtol=1e-5)


def test_periods_per_year_changes_rf_conversion():
    # With periods_per_year=252 the annual rf must be de-compounded to a DAILY
    # rate, not a monthly one.
    returns = [0.001, -0.001, 0.001]
    daily_rf = 1.05 ** (1 / 252) - 1
    mean, std = np.mean(returns), np.std(returns, ddof=1)
    np.testing.assert_allclose(
        sharpe_ratio(returns, annual_rf=0.05, periods_per_year=252),
        (mean - daily_rf) / std,
        rtol=1e-9,
    )


def test_annualize_scales_by_sqrt_periods():
    returns = [0.001, -0.001, 0.001]
    per_period = sharpe_ratio(returns, annual_rf=0.0, periods_per_year=252)
    annualized = sharpe_ratio(returns, annual_rf=0.0, periods_per_year=252, annualize=True)
    np.testing.assert_allclose(annualized, per_period * math.sqrt(252), rtol=1e-9)


def test_defaults_unchanged_monthly_not_annualized():
    # Default call must still be the monthly, NON-annualized Sharpe.
    returns = [0.10, -0.10, 0.10]
    np.testing.assert_allclose(sharpe_ratio(returns, annual_rf=0.0), 0.2886751, rtol=1e-5)
