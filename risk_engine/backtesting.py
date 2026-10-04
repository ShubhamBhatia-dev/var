"""
risk_engine/backtesting.py
--------------------------
VaR backtesting orchestration module.

Combines rolling VaR/ES calculation with statistical tests.

BACKTESTING OVERVIEW:
=====================
VaR backtesting is the process of comparing ex-ante (forecasted) VaR
estimates against ex-post (realized) trading outcomes.

The backtesting procedure:
1. For each date t, calculate VaR using returns from [t-window, t-1].
2. Record the realized P&L on date t.
3. Flag a violation if realized P&L < -VaR.
4. Count violations over the backtesting period.
5. Apply statistical tests to the violation sequence.

REGULATORY CONTEXT:
===================
Under Basel II/III, banks are required to backtest their internal models.
The Basel "Traffic Light" system categorizes banks by exception count:
  Green zone: 0-4 exceptions in 250 trading days → model accepted
  Yellow zone: 5-9 exceptions → increased scrutiny / capital add-on
  Red zone: 10+ exceptions → model likely rejected

This engine implements the same statistical framework but is NOT a
regulatory-certified implementation. It is an educational demonstration
of quantitative backtesting methodology.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

from .kupiec import kupiec_pof_test
from .christoffersen import christoffersen_independence_test, conditional_coverage_test

logger = logging.getLogger(__name__)


def run_backtest(
    rolling_results: pd.DataFrame,
    confidence_level: float = 0.99,
    alpha: float = 0.05,
) -> dict:
    """
    Run complete VaR backtesting analysis.

    Parameters
    ----------
    rolling_results : pd.DataFrame
        Output from historical_simulation.compute_rolling_var_es().
        Must have columns: VaR_1D, Realized_PnL, Violation.
    confidence_level : float
        VaR confidence level used (e.g., 0.99).
    alpha : float
        Statistical test significance level (default: 0.05).

    Returns
    -------
    dict
        Complete backtesting results including:
        - Summary statistics
        - Kupiec POF test results
        - Christoffersen Independence test results
        - Conditional Coverage test results
        - Violation detail records
        - Basel traffic light zone
    """
    violations_series = rolling_results["Violation"].values
    n_obs = int(len(violations_series))
    n_violations = int(violations_series.sum())
    p_theoretical = 1.0 - confidence_level

    # Basic statistics
    violation_rate = n_violations / n_obs if n_obs > 0 else 0.0
    expected_violations = n_obs * p_theoretical

    # Identify violation dates
    violation_mask = rolling_results["Violation"] == 1
    violation_records = []
    for date, row in rolling_results[violation_mask].iterrows():
        violation_records.append({
            "date": str(date.date()),
            "realized_pnl": float(row["Realized_PnL"]),
            "var_1d": float(row["VaR_1D"]),
            "loss_in_excess_of_var": float(-row["Realized_PnL"] - row["VaR_1D"]),
        })

    # Run statistical tests
    kupiec_result = kupiec_pof_test(
        n_observations=n_obs,
        n_violations=n_violations,
        confidence_level=confidence_level,
        alpha=alpha,
    )

    independence_result = christoffersen_independence_test(
        violations=violations_series.tolist(),
        alpha=alpha,
    )

    cc_result = conditional_coverage_test(
        violations=violations_series.tolist(),
        n_observations=n_obs,
        confidence_level=confidence_level,
        alpha=alpha,
    )

    # Basel traffic light (for reference only — not regulatory)
    # Based on 250-observation equivalent scaling
    scaled_violations = n_violations * (250 / n_obs) if n_obs > 0 else 0
    if scaled_violations <= 4:
        basel_zone = "GREEN"
    elif scaled_violations <= 9:
        basel_zone = "YELLOW"
    else:
        basel_zone = "RED"

    backtest_result = {
        "summary": {
            "n_observations": n_obs,
            "n_violations": n_violations,
            "expected_violations": float(expected_violations),
            "violation_rate": float(violation_rate),
            "expected_violation_rate": float(p_theoretical),
            "confidence_level": confidence_level,
            "backtest_date_start": str(rolling_results.index[0].date()),
            "backtest_date_end": str(rolling_results.index[-1].date()),
            "basel_zone_reference": basel_zone,
            "scaled_violations_250d": float(scaled_violations),
        },
        "kupiec_pof": kupiec_result,
        "christoffersen_independence": independence_result,
        "conditional_coverage": cc_result,
        "violations": violation_records,
    }

    logger.info(
        f"Backtest complete: N={n_obs}, violations={n_violations} "
        f"({violation_rate:.2%}), Basel zone: {basel_zone}"
    )

    return backtest_result


def get_violation_timeline(rolling_results: pd.DataFrame) -> list:
    """
    Generate the violation timeline for dashboard visualization.

    Parameters
    ----------
    rolling_results : pd.DataFrame
        Rolling backtesting results.

    Returns
    -------
    list of dict
        One entry per observation with date, P&L, VaR, and violation flag.
    """
    records = []
    for date, row in rolling_results.iterrows():
        records.append({
            "date": str(date.date()),
            "realized_pnl": float(row["Realized_PnL"]),
            "var_1d": float(row["VaR_1D"]),
            "es_1d": float(row["ES_1D"]),
            "violation": int(row["Violation"]),
        })
    return records
