"""
risk_engine/var.py
------------------
Value at Risk (VaR) calculation module.

DEFINITION:
===========
Value at Risk (VaR) at confidence level α is defined as the loss
that is NOT exceeded with probability α.

Formally:
    P(Loss > VaR_α) = 1 - α

For 99% VaR: P(Loss > VaR_99) = 1% = 0.01

HISTORICAL SIMULATION VaR:
===========================
We use the EMPIRICAL quantile of the P&L distribution.

Steps:
1. Sort all hypothetical P&Ls from worst to best.
2. VaR is the loss at the 1st percentile (for 99% confidence).

Convention: VaR is expressed as a POSITIVE number (loss magnitude).

If the 1st percentile of the P&L distribution is -$250,000:
  VaR_99 = $250,000 (we lose at least this much 1% of the time)

Mathematical formulation:
  VaR_99 = -quantile(P&L_distribution, 0.01)

The negative sign converts the negative P&L quantile to a positive loss figure.

SQUARE-ROOT-OF-TIME APPROXIMATION:
===================================
The square-root-of-time rule states:
  VaR_10d ≈ VaR_1d × sqrt(10)

This approximation relies on the assumption that:
1. Returns are i.i.d. (independent and identically distributed)
2. Returns are normally distributed (or at least have finite variance)

These assumptions are generally VIOLATED in real financial returns:
- Returns exhibit autocorrelation (momentum, mean-reversion)
- Returns have fat tails (leptokurtosis)
- Volatility clusters over time (GARCH effects)

Therefore, we use explicit 10-day historical scenarios as the PRIMARY method
and show the sqrt(10) approximation only as a secondary informational metric.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def calculate_var(
    pnl: np.ndarray,
    confidence: float = 0.99,
) -> float:
    """
    Calculate Historical Simulation VaR from a P&L distribution.

    VaR convention: Returned as a POSITIVE number representing a loss.

    The VaR is the negative of the empirical (1-confidence) quantile.
    For 99% confidence, this is the negative of the 1st percentile.

    Parameters
    ----------
    pnl : np.ndarray
        Array of hypothetical portfolio P&L values (can be positive or negative).
    confidence : float
        VaR confidence level (e.g., 0.99 for 99%).

    Returns
    -------
    float
        VaR as a positive loss amount in the same units as `pnl`.

    Example
    -------
    >>> pnl = np.array([-300, -200, -150, -50, 100, 200, ...])
    >>> var = calculate_var(pnl, confidence=0.99)
    # Returns positive number e.g., 250000
    """
    if len(pnl) == 0:
        logger.warning("Empty P&L array passed to calculate_var. Returning 0.")
        return 0.0

    # tail_probability = 1 - confidence = 0.01 for 99% VaR
    tail_probability = 1.0 - confidence

    # np.quantile returns the (tail_probability * 100)th percentile
    # For 99% VaR, this is the 1st percentile — the boundary of the worst 1%
    pnl_quantile = float(np.quantile(pnl, tail_probability))

    # VaR is expressed as a positive loss:
    # If the 1st percentile P&L = -$250,000, then VaR = $250,000
    var = -pnl_quantile

    logger.debug(
        f"VaR calculation: confidence={confidence:.2%}, "
        f"tail_prob={tail_probability:.2%}, "
        f"P&L quantile={pnl_quantile:,.0f}, "
        f"VaR={var:,.0f}"
    )

    return max(var, 0.0)  # VaR should never be negative (that would imply guaranteed gains)


def calculate_var_series(
    pnl_series: pd.Series,
    confidence: float = 0.99,
) -> float:
    """
    Wrapper for pandas Series input.

    Parameters
    ----------
    pnl_series : pd.Series
        P&L series.
    confidence : float
        Confidence level.

    Returns
    -------
    float
        VaR as positive loss.
    """
    return calculate_var(pnl_series.dropna().values, confidence)


def calculate_var_sqrt_of_time(
    var_1d: float,
    holding_period: int = 10,
) -> float:
    """
    Square-root-of-time approximation for multi-day VaR.

    IMPORTANT: This is an APPROXIMATION only.
    It assumes returns are i.i.d. and variance scales linearly with time.

    VaR_h ≈ VaR_1d × sqrt(h)

    This is provided as a secondary informational metric.
    The primary 10-day VaR uses explicit historical 10-day scenarios.

    Mathematical basis:
      If daily P&L ~ N(0, σ²) and returns are i.i.d.:
        Var(h-day P&L) = h × σ²
        std(h-day P&L) = σ × sqrt(h)
        VaR_h = VaR_1 × sqrt(h)

    Limitations:
      - Fat tails make this underestimate tail risk.
      - Autocorrelation makes this either over- or under-estimate.
      - Volatility clustering violates the i.i.d. assumption.

    Parameters
    ----------
    var_1d : float
        1-day VaR (positive loss amount).
    holding_period : int
        Holding period in days.

    Returns
    -------
    float
        Approximate multi-day VaR (positive loss amount).
        LABELED AS APPROXIMATION — not the primary estimate.
    """
    approx_var = var_1d * np.sqrt(holding_period)
    logger.debug(
        f"Sqrt-of-time approx: VaR_1d={var_1d:,.0f}, "
        f"h={holding_period}, VaR_h_approx={approx_var:,.0f}"
    )
    return approx_var
