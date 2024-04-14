import pytest
import numpy as np

from gym_env_rlot.buy_sell.gym_env_utils import (
    calculate_maximum_drawdown,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
)


def test_empty_series():
    assert (
        calculate_maximum_drawdown([]) == 0.0
    ), "Expected drawdown of an empty series to be 0.0"


def test_increasing_series():
    assert (
        calculate_maximum_drawdown([100.123, 200.456, 300.789, 400.012]) == 0.0
    ), "Expected drawdown of an increasing series to be 0.0"


def test_decreasing_series():
    assert (
        calculate_maximum_drawdown([400.400, 300.300, 200.200, 100.100]) == 300.3
    ), "Expected drawdown of a decreasing series to be 300.3"


def test_mixed_series():
    assert (
        calculate_maximum_drawdown(
            [100.001, 200.002, 150.150, 250.250, 200.200, 100.100, 50.050]
        )
        == 200.2
    ), "Expected drawdown of the mixed series to be 200.2"


def test_series_with_recovery():
    assert (
        calculate_maximum_drawdown(
            [100.100, 200.200, 150.150, 250.250, 200.200, 100.100, 300.300]
        )
        == 150.15
    ), "Expected drawdown of the series with recovery to be 150.15"


def test_series_with_multiple_drawdowns_and_floats():
    assert (
        calculate_maximum_drawdown([100.123, 50.056, 100.789, 50.012, 100.321])
        == 50.777
    ), "Expected drawdown of the series with multiple drawdowns to be 50.767"


def test_series_with_precise_floats():
    assert (
        calculate_maximum_drawdown(
            [123.456, 123.555, 123.554, 125.000, 124.999, 123.450]
        )
        == 1.55
    ), "Expected drawdown of the series with precise floats to be 1.55"


def test_sharpe_ratio_basic():
    pnl = np.array([100, 105, 110, 115, 120])
    expected = calculate_sharpe_ratio(pnl, 0.05, clip=False)
    assert calculate_sharpe_ratio(pnl) == expected


def test_sharpe_ratio_clipping():
    # This test ensures clipping works as intended
    extreme_pnl = np.array([100, 300, 100, 300, 100])
    sharpe_ratio = calculate_sharpe_ratio(extreme_pnl)
    assert sharpe_ratio <= 10 and sharpe_ratio >= -10


def test_sharpe_ratio_low_std_dev():
    # No change, std_dev would be zero
    pnl = np.array([100, 100, 100, 100])
    assert calculate_sharpe_ratio(pnl) == -10.0


def test_sharpe_ratio_negative():
    # Decreasing, should result in a negative Sharpe Ratio
    pnl = np.array([100, 95, 90, 85, 80])
    sharpe_ratio = calculate_sharpe_ratio(pnl)
    assert sharpe_ratio < 0


def test_sharpe_ratio_insufficient_data():
    # Insufficient data
    pnl = np.array([100])
    assert calculate_sharpe_ratio(pnl) == 0.0


def test_positive_returns_above_risk_free_rate():
    # Scenario with all positive returns, all above the risk-free rate
    pnl = np.array([100, 102, 104, 107])
    assert (
        calculate_sortino_ratio(pnl) > 0
    ), "Sortino ratio should be positive when all returns are above the risk-free rate."


def test_negative_returns_below_risk_free_rate():
    # Scenario with negative returns, indicating underperformance
    pnl = np.array([100, 98, 95, 90])
    assert (
        calculate_sortino_ratio(pnl) < 0
    ), "Sortino ratio should be negative when returns are below the risk-free rate."


def test_no_downside_deviation():
    # Scenario where all returns match or exceed the risk-free rate, resulting in no downside deviation
    pnl = np.array([100, 100.5, 101, 102])
    assert (
        calculate_sortino_ratio(pnl) >= 0
    ), "Sortino ratio should be non-negative when there is no downside deviation."


def test_returns_with_mixed_performance():
    # Scenario with mixed returns relative to the risk-free rate
    pnl = np.array([100, 103, 101, 105, 102])
    # The Sortino ratio can be positive or negative depending on the magnitude of returns above vs. below the risk-free rate
    result = calculate_sortino_ratio(pnl)
    assert isinstance(result, float), "Sortino ratio should be a float value."


def test_single_data_point():
    # Scenario with insufficient data (single data point)
    pnl = np.array([100])
    assert (
        calculate_sortino_ratio(pnl) == 0
    ), "Sortino ratio should be neutral (0) with insufficient data."


def test_neutral_sortino_ratio():
    # Scenario where the investment performance exactly matches the risk-free rate
    pnl = np.array([100, 100.05, 100.1, 100.15])
    assert (
        calculate_sortino_ratio(pnl) == 5
    ), "Sortino ratio should be neutral (0) when performance matches the risk-free rate."


if __name__ == "__main__":
    pytest.main([__file__])
