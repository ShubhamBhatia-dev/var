"""
risk_engine/historical_simulation.py
-------------------------------------
Historical Simulation engine for portfolio P&L generation.

METHODOLOGY:
===========
Historical Simulation (HS) is a non-parametric VaR methodology.
Rather than assuming a particular distribution for returns, it uses
the empirical distribution of historical risk factor changes.

Key principle: We assume that historical return patterns will repeat.

For each historical scenario (trading day t):
1. Record the actual historical log returns across all risk factors.
2. Apply these SAME returns to the CURRENT portfolio positions.
3. Calculate the hypothetical portfolio P&L under that historical scenario.

Crucial design property: Joint historical movements are preserved.
If on date t we observed:
  SPY:    -3.2%
  QQQ:    -4.1%
  EURUSD: +0.8% (USD weakened)
  IEF:    +1.2% (flight to quality)

These shocks are applied TOGETHER to the current portfolio.
This preserves historical correlations — a key advantage over methods
that independently sample each risk factor.

P&L CALCULATION:
================
For each historical scenario t and asset i:

  position_value_i = weight_i × portfolio_notional

  P&L from asset i = position_value_i × (exp(r_{i,t}) - 1)

where r_{i,t} is the log return of asset i on historical day t.

Total portfolio P&L for scenario t:
  P&L_t = sum_i [ position_value_i × (exp(r_{i,t}) - 1) ]

This is a delta-equivalent approximation (full revaluation for linear assets).
For equity ETFs and FX spot: this is exact (linear payoff).
For bond ETFs: this is a good approximation since ETFs are linear instruments.
For nonlinear instruments (options): this would require full revaluation.

10-DAY SCENARIOS:
=================
For 10-day VaR, we use OVERLAPPING 10-day windows.

For each starting date t:
  10-day log return of asset i = sum_{k=0}^{9} r_{i, t+k}

We then apply these 10-day cumulative returns to the current portfolio.

Note: Overlapping windows introduce serial correlation in the scenario
distribution. Non-overlapping windows would be statistically cleaner but
would drastically reduce the sample size (from ~500 to ~50 scenarios).

ASSUMPTION DOCUMENTATION:
=========================
1. Positions are fixed in dollar terms during the holding period.
2. Log returns are additive for multi-day cumulation.
3. No intra-day rebalancing is assumed.
4. All positions are mark-to-market (no accrual accounting).
"""

import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from .portfolio import Portfolio

logger = logging.getLogger(__name__)


def compute_portfolio_pnl_1d(
    log_returns: pd.DataFrame,
    portfolio: Portfolio,
    window: int = 500,
) -> pd.Series:
    """
    Compute 1-day hypothetical portfolio P&L for all historical scenarios.

    Uses the most recent `window` daily log returns as scenarios.

    For each historical day t in the window:
      P&L_t = sum_i [position_i × (exp(r_{i,t}) - 1)]

    Parameters
    ----------
    log_returns : pd.DataFrame
        Daily log returns, one column per asset.
    portfolio : Portfolio
        Portfolio definition with weights and notional.
    window : int
        Number of historical scenarios (days) to use.

    Returns
    -------
    pd.Series
        Hypothetical P&L for each scenario (indexed by scenario date).
    """
    # Select only assets in the portfolio that have return data
    assets = [a for a in portfolio.assets if a in log_returns.columns]
    if len(assets) < len(portfolio.assets):
        missing = set(portfolio.assets) - set(assets)
        logger.warning(f"Return data missing for assets: {missing}. Excluding from P&L.")

    # Take the most recent `window` scenarios
    scenarios = log_returns[assets].tail(window)

    # Position values in USD for each asset
    pos_values = portfolio.position_values()
    position_array = np.array([pos_values[a] for a in assets])

    # Convert log returns to simple returns: exp(r) - 1
    # Then multiply by position value to get dollar P&L
    # P&L_t = position_i * (exp(r_{i,t}) - 1) for all i, summed
    simple_returns = np.expm1(scenarios.values)  # expm1(x) = exp(x) - 1 (numerically stable)
    pnl_array = simple_returns @ position_array  # dot product: scenarios × assets

    pnl = pd.Series(pnl_array, index=scenarios.index, name="Portfolio_PnL_1D")

    logger.info(
        f"1-day P&L distribution computed: {len(pnl)} scenarios, "
        f"min={pnl.min():,.0f}, max={pnl.max():,.0f}"
    )

    return pnl


def compute_portfolio_pnl_10d(
    log_returns: pd.DataFrame,
    portfolio: Portfolio,
    window: int = 500,
    holding_period: int = 10,
) -> pd.Series:
    """
    Compute 10-day hypothetical portfolio P&L using overlapping windows.

    For each starting date t, we sum the next `holding_period` log returns
    to create a multi-day scenario, then apply to the portfolio.

    This is NOT the square-root-of-time approximation.
    We explicitly construct multi-day scenarios from actual return sequences,
    preserving the autocorrelation structure in historical data.

    For starting date t:
      r_{i, t:t+h} = sum_{k=0}^{h-1} r_{i, t+k}

      P&L_t = sum_i [position_i × (exp(r_{i, t:t+h}) - 1)]

    Parameters
    ----------
    log_returns : pd.DataFrame
        Daily log returns.
    portfolio : Portfolio
        Portfolio definition.
    window : int
        Number of 10-day scenarios to use.
    holding_period : int
        Number of days to cumulate (default: 10).

    Returns
    -------
    pd.Series
        10-day hypothetical P&L for each scenario.
    """
    assets = [a for a in portfolio.assets if a in log_returns.columns]
    pos_values = portfolio.position_values()
    position_array = np.array([pos_values[a] for a in assets])

    returns_array = log_returns[assets].values
    n_total = len(returns_array)

    # Build overlapping holding_period-day cumulative return scenarios
    # For each starting index i, sum returns from i to i + holding_period - 1
    max_start = n_total - holding_period
    if max_start < window:
        logger.warning(
            f"Only {max_start} valid 10-day scenarios available. "
            f"Requested: {window}. Using all available."
        )
        n_scenarios = max_start
    else:
        n_scenarios = window

    # Use the most recent `n_scenarios` starting dates
    start_indices = range(n_total - holding_period - n_scenarios + 1, n_total - holding_period + 1)
    scenario_dates = log_returns.index[list(start_indices)]

    cumulative_returns = np.zeros((n_scenarios, len(assets)))
    for idx, start in enumerate(start_indices):
        # Sum log returns across the holding period (additive for log returns)
        cumulative_returns[idx, :] = returns_array[start : start + holding_period, :].sum(axis=0)

    # Convert cumulative log returns to dollar P&L
    simple_returns = np.expm1(cumulative_returns)
    pnl_array = simple_returns @ position_array

    pnl = pd.Series(pnl_array, index=scenario_dates, name="Portfolio_PnL_10D")

    logger.info(
        f"10-day P&L distribution computed: {len(pnl)} scenarios, "
        f"min={pnl.min():,.0f}, max={pnl.max():,.0f}"
    )

    return pnl


def compute_rolling_scenarios(
    log_returns: pd.DataFrame,
    portfolio: Portfolio,
    window: int = 500,
    min_window: int = 250,
) -> pd.DataFrame:
    """
    Compute rolling 1-day portfolio P&L for each historical trading day.

    For each date t in the backtesting period:
    - Uses returns from t-window to t-1 as scenarios (NO look-ahead).
    - Computes VaR applicable to day t.
    - Records realized P&L on day t (i.e., the actual observed P&L).

    This is the rolling backtesting engine.

    Visualization of the look-ahead-safe design:
      Historical window          Forecast  Realized
      [t-500 ... t-1]     →       VaR_t     P&L_t

    We NEVER use P&L from day t when estimating VaR for day t.

    Parameters
    ----------
    log_returns : pd.DataFrame
        All available log returns.
    portfolio : Portfolio
        Portfolio definition.
    window : int
        Rolling window size.
    min_window : int
        Minimum observations before we start generating forecasts.

    Returns
    -------
    pd.DataFrame
        Columns: Date, Realized_PnL, plus the full scenario distribution
        is returned for VaR/ES calculation in var.py.
        Actually returns: index=Date, columns=[Realized_PnL]
        VaR is calculated separately in var.py using this rolling framework.
    """
    assets = [a for a in portfolio.assets if a in log_returns.columns]
    pos_values = portfolio.position_values()
    position_array = np.array([pos_values[a] for a in assets])

    returns_array = log_returns[assets].values
    dates = log_returns.index
    n_total = len(returns_array)

    realized_pnls = []
    forecast_dates = []

    for t in range(min_window, n_total):
        # Realized P&L on day t: apply day-t returns to portfolio
        # This is what actually happened on that trading day
        realized_return = np.expm1(returns_array[t, :])
        realized_pnl = float(np.dot(realized_return, position_array))

        realized_pnls.append(realized_pnl)
        forecast_dates.append(dates[t])

    result = pd.DataFrame(
        {"Realized_PnL": realized_pnls},
        index=forecast_dates,
    )

    logger.info(f"Rolling scenarios computed: {len(result)} forecast dates")

    return result


def compute_rolling_var_es(
    log_returns: pd.DataFrame,
    portfolio: Portfolio,
    window: int = 500,
    confidence: float = 0.99,
    min_window: int = 250,
) -> pd.DataFrame:
    """
    Compute rolling daily VaR and ES for the entire backtesting period.

    For each date t (starting after `min_window` observations):
      - Historical window: returns[t-window:t] (strictly before t)
      - Compute hypothetical P&Ls for the window
      - Estimate VaR as -quantile(P&L, 1-confidence)
      - Estimate ES as -mean(P&L[P&L <= VaR threshold])

    NO LOOK-AHEAD BIAS: Day t's VaR uses ONLY returns from t-window to t-1.
    The realized P&L on day t is then compared against this VaR estimate.

    Parameters
    ----------
    log_returns : pd.DataFrame
        All available log returns.
    portfolio : Portfolio
        Portfolio definition.
    window : int
        Rolling window size (number of historical scenarios).
    confidence : float
        VaR confidence level (e.g., 0.99 for 99%).
    min_window : int
        Minimum observations before starting.

    Returns
    -------
    pd.DataFrame
        Columns: Date (index), VaR_1D, ES_1D, Realized_PnL, Violation
    """
    from .var import calculate_var
    from .expected_shortfall import calculate_es

    assets = [a for a in portfolio.assets if a in log_returns.columns]
    pos_values = portfolio.position_values()
    position_array = np.array([pos_values[a] for a in assets])

    returns_array = log_returns[assets].values
    dates = log_returns.index
    n_total = len(returns_array)

    rows = []

    for t in range(min_window, n_total):
        # Historical window: t-window to t-1 (no day t!)
        start_idx = max(0, t - window)
        historical_returns = returns_array[start_idx:t, :]

        # Compute hypothetical P&L for each historical scenario
        simple_ret = np.expm1(historical_returns)
        scenario_pnls = simple_ret @ position_array

        # Compute VaR and ES
        var_1d = calculate_var(scenario_pnls, confidence)
        es_1d = calculate_es(scenario_pnls, confidence)

        # Realized P&L on day t
        realized_return = np.expm1(returns_array[t, :])
        realized_pnl = float(np.dot(realized_return, position_array))

        # Violation: realized loss exceeds VaR
        # Breach occurs when realized P&L < -VaR (VaR is positive)
        violation = int(realized_pnl < -var_1d)

        rows.append(
            {
                "VaR_1D": var_1d,
                "ES_1D": es_1d,
                "Realized_PnL": realized_pnl,
                "Violation": violation,
                "N_Scenarios": len(scenario_pnls),
            }
        )

    result = pd.DataFrame(rows, index=dates[min_window:])
    result.index.name = "Date"

    logger.info(
        f"Rolling VaR/ES computed: {len(result)} dates, "
        f"{int(result['Violation'].sum())} violations"
    )

    return result
