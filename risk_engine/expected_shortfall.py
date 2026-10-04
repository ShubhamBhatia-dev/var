"""
risk_engine/expected_shortfall.py
----------------------------------
Expected Shortfall (ES) — also called Conditional VaR (CVaR) or
Average Value at Risk (AVaR).

DEFINITION:
===========
Expected Shortfall at confidence level α is the EXPECTED LOSS conditional
on the loss exceeding the VaR threshold.

Formally:
  ES_α = -E[P&L | P&L ≤ VaR_α_percentile]
       = -E[P&L | P&L ≤ quantile(P&L, 1-α)]

For 99% ES: Average of the worst 1% of outcomes.

WHY ES MATTERS:
===============
VaR answers: "What is the minimum loss in the worst 1% of cases?"
ES answers:  "Given that we ARE in the worst 1% of cases, what is
              the average loss?"

ES has better theoretical properties:
1. SUBADDITIVITY: ES(A + B) ≤ ES(A) + ES(B)
   → Diversification always reduces ES (not always true for VaR)
2. COHERENCE: ES satisfies all coherent risk measure axioms
3. TAIL SENSITIVITY: ES is sensitive to the shape of the tail beyond VaR

Example:
  P&L distribution's 1st percentile: -$250,000 → VaR = $250,000
  Average loss in the worst 1%:      -$380,000 → ES  = $380,000

  The ES tells us that when things go wrong (beyond VaR), we typically
  lose $380,000 — not just the VaR threshold of $250,000.

ROBUSTNESS NOTE:
================
When the sample is small (e.g., 500 observations, 1% tail = 5 observations),
ES can be noisy. We handle this gracefully.

Modern regulatory frameworks (Basel III/IV, FRTB) use ES at 97.5% confidence
rather than VaR at 99%, precisely because ES captures tail risk better.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def calculate_es(
    pnl: np.ndarray,
    confidence: float = 0.99,
) -> float:
    """
    Calculate Expected Shortfall (ES) from a P&L distribution.

    ES is returned as a POSITIVE number representing the expected loss
    in the worst (1 - confidence) fraction of outcomes.

    Algorithm:
    1. Find the VaR threshold = quantile(P&L, 1 - confidence)
    2. Select all P&L values AT OR BELOW this threshold
    3. ES = -mean(selected P&L values)

    The negative sign converts the negative P&L to a positive loss figure.

    Parameters
    ----------
    pnl : np.ndarray
        Array of hypothetical portfolio P&L values.
    confidence : float
        ES confidence level (e.g., 0.99 for 99%).

    Returns
    -------
    float
        ES as a positive loss amount.

    Notes
    -----
    The choice of "≤ threshold" vs "< threshold" affects results slightly.
    Using "≤" includes the VaR boundary itself, which is standard practice
    in historical simulation.
    """
    if len(pnl) == 0:
        logger.warning("Empty P&L array passed to calculate_es. Returning 0.")
        return 0.0

    tail_probability = 1.0 - confidence
    var_threshold = float(np.quantile(pnl, tail_probability))

    # Select all P&L values in the tail (at or below the VaR threshold)
    tail_pnl = pnl[pnl <= var_threshold]

    if len(tail_pnl) == 0:
        # This can happen if all P&L values are above the threshold
        logger.warning("No tail observations found for ES calculation. Using VaR threshold.")
        return -var_threshold

    # ES = average loss in the tail
    es = -float(np.mean(tail_pnl))

    # ES should always be >= VaR (tail average >= tail threshold)
    # If ES < VaR, it indicates a numerical issue
    var = -var_threshold
    if es < var:
        logger.warning(
            f"ES ({es:,.0f}) < VaR ({var:,.0f}). This may indicate insufficient "
            "tail observations. Using VaR as ES lower bound."
        )
        es = var

    n_tail_obs = len(tail_pnl)
    logger.debug(
        f"ES calculation: confidence={confidence:.2%}, "
        f"VaR_threshold={var_threshold:,.0f}, "
        f"tail_obs={n_tail_obs}, ES={es:,.0f}"
    )

    return es


def calculate_es_series(
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
        ES as positive loss.
    """
    return calculate_es(pnl_series.dropna().values, confidence)


def es_decomposition(
    pnl: np.ndarray,
    confidence: float = 0.99,
) -> dict:
    """
    Return detailed ES decomposition for reporting.

    Parameters
    ----------
    pnl : np.ndarray
        P&L distribution.
    confidence : float
        Confidence level.

    Returns
    -------
    dict
        ES breakdown with VaR, ES, tail statistics.
    """
    from .var import calculate_var

    var = calculate_var(pnl, confidence)
    es = calculate_es(pnl, confidence)

    tail_probability = 1.0 - confidence
    var_threshold = float(np.quantile(pnl, tail_probability))
    tail_pnl = pnl[pnl <= var_threshold]

    return {
        "var": var,
        "es": es,
        "es_to_var_ratio": es / var if var != 0 else None,
        "n_tail_observations": int(len(tail_pnl)),
        "tail_probability": tail_probability,
        "worst_loss": float(-pnl.min()) if len(pnl) > 0 else None,
        "tail_std": float(tail_pnl.std()) if len(tail_pnl) > 1 else None,
    }
