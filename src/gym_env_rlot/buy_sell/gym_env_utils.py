import numpy as np


def calculate_maximum_drawdown(cumulative_pnl):
    """Calculate the maximum drawdown of a P&L series."""
    cumulative_pnl = np.array(cumulative_pnl, dtype=np.float64)
    if len(cumulative_pnl) == 0:
        maximum_drawdown = 0.0
    else:
        peak = cumulative_pnl[0]
        maximum_drawdown = 0.0

        for pnl in cumulative_pnl:
            if pnl > peak:
                peak = pnl
            drawdown = peak - pnl
            if drawdown > maximum_drawdown:
                maximum_drawdown = drawdown

    return -np.round(maximum_drawdown, 6)


def calculate_sharpe_ratio(cumulative_pnl, risk_free_rate=0.05, clip=True):
    """Calculate the Sharpe ratio of a P&L series.

    A good Sharpe Ratio typically is above 1, indicating that the investment
    returns exceed the risk taken (higher returns relative to volatility).
    A Sharpe Ratio between 1 and 2 is considered acceptable to good, and values
    above 2 are often seen as very good.

    A bad Sharpe Ratio is one that is less than 1, suggesting that the investment's
    returns do not adequately compensate for the risk. A Sharpe Ratio near 0 or
    negative indicates poor performance, where the risk-free rate may outperform
    the investment when adjusted for volatility.
    """
    cumulative_pnl = np.array(cumulative_pnl, dtype=np.float64)
    if len(cumulative_pnl) <= 2:
        sharpe_ratio = 0.0
    else:
        returns = np.diff(cumulative_pnl) / (cumulative_pnl[:-1] + np.finfo(float).eps)
        mean_return = np.mean(returns)
        std_dev = np.std(returns)

        # Avoid division by zero
        if std_dev == 0:
            std_dev += np.finfo(np.float64).eps

        sharpe_ratio = (mean_return - risk_free_rate) / std_dev

    if clip:
        sharpe_ratio = np.clip(sharpe_ratio, -3, 3)

    return np.round(sharpe_ratio, 3)


def calculate_sortino_ratio(cumulative_pnl, risk_free_rate=0.05, default_high_value=5):
    """
    Calculate the Sortino Ratio for a given series of returns.

    A good Sortino Ratio is generally above 2, indicating that the investment generates
    a high return on its downside risk. Values above 2 are considered very good,
    reflecting that the investment has strong performance relative to the negative
    volatility (downside risk) it experiences.

    A bad Sortino Ratio is one that is close to 0, negative, or significantly below 1.
    This suggests that the investment's returns are not sufficient to justify its
    downside risk. Unlike the Sharpe Ratio, the Sortino Ratio only considers downside
    volatility, making positive values below 1 less desirable but not as critical if
    the downside risk is low relative to returns.
    """
    cumulative_pnl = np.array(cumulative_pnl, dtype=np.float64)
    if len(cumulative_pnl) <= 2:
        sortino_ratio = 0.0
    else:

        # Calculate returns from cumulative P&L
        returns = np.diff(cumulative_pnl) / (cumulative_pnl[:-1] + np.finfo(float).eps)

        hourly_risk_free_rate = (1 + risk_free_rate) ** (1 / (5.5 * 252)) - 1
        # Calculate excess returns
        excess_returns = returns - hourly_risk_free_rate

        # Calculate downside deviation
        downside_returns = excess_returns[excess_returns < 0]
        if downside_returns.size == 0:
            sortino_ratio = default_high_value
        else:
            downside_deviation = np.sqrt(np.mean(downside_returns**2))

            # Calculate mean excess return
            mean_excess_return = np.mean(excess_returns)

            # Calculate the Sortino Ratio
            sortino_ratio = mean_excess_return / downside_deviation

            sortino_ratio = np.clip(sortino_ratio, -1, 100)

    return np.round(sortino_ratio, 3)


def calculate_calmar_ratio(cumulative_pnl, risk_free_rate=0.05, clip_range=(-100, 100)):
    """
    Calculate and clip the Calmar Ratio of a cumulative P&L series.

    The Calmar Ratio is a performance metric that evaluates the risk-adjusted return
    of an investment strategy, focusing on the risk of significant losses as captured
    by the maximum drawdown. It is defined as the ratio of the annualized excess return
    over the maximum drawdown.

    Parameters:
    - cumulative_pnl (np.array): An array of cumulative profit and loss values.
    - risk_free_rate (float, optional): The annual risk-free rate, defaulting to 5%.
    - clip_range (tuple, optional): The range (min, max) to clip the Calmar Ratio values,
                                    defaulting to (-10, 10).

    Returns:
    - float: The clipped Calmar Ratio, rounded to 3 decimal places.

    A good Calmar Ratio, typically above 2, indicates strong performance relative to
    the downside risk. Values significantly above this range are considered exceptional.
    Negative or low positive values suggest underperformance relative to the downside risk.
    The function clips extreme values to the specified range to prevent outliers from
    distorting overall performance assessments.
    """
    if len(cumulative_pnl) <= 1:
        calmar_ratio = 0.0  # Not enough data to compute the ratio.
    else:
        total_return = (
            cumulative_pnl[-1] / (cumulative_pnl[0] + np.finfo(float).eps) - 1
        )
        # Adjust the calculation to avoid overflow
        try:
            if total_return <= -1:
                # Consider using -np.inf to indicate extreme loss
                annualized_return = -np.inf

            else:
                years = len(cumulative_pnl) / 252.0 * 5.5
                annualized_return = (1 + total_return) ** (1 / years) - 1
        except OverflowError:
            annualized_return = np.inf if total_return > 0 else -np.inf

        max_drawdown = calculate_maximum_drawdown(cumulative_pnl)
        if max_drawdown <= 0:
            max_drawdown = np.finfo(float).eps  # Avoid division by zero.

        calmar_ratio = (
            annualized_return / max_drawdown
            if max_drawdown != np.finfo(float).eps
            else np.inf
        )
        calmar_ratio = np.clip(calmar_ratio, clip_range[0], clip_range[1])

    return np.round(calmar_ratio, 3)
