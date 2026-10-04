"""
risk_engine/kupiec.py
---------------------
Kupiec Proportion of Failures (POF) Test.

BACKGROUND:
===========
The Kupiec POF test (1995) is a statistical test of VaR model accuracy.
It tests whether the PROPORTION of VaR exceptions is consistent with the
model's stated confidence level.

NULL HYPOTHESIS:
================
H0: The proportion of VaR exceptions equals the theoretical violation
    probability p = 1 - confidence (e.g., p = 0.01 for 99% VaR).

The test does NOT claim to prove a model is "right" or "wrong."
It assesses whether observed exception frequency is statistically
inconsistent with the assumed coverage probability at a chosen
significance level.

LIKELIHOOD RATIO STATISTIC:
============================
Under H0 (assumed violation prob = p):
  L(H0) = (1-p)^(N-x) × p^x

Under H1 (estimated violation prob = p_hat = x/N):
  L(H1) = (1-p_hat)^(N-x) × p_hat^x

Likelihood Ratio statistic:
  LR_POF = -2 × [ln L(H0) - ln L(H1)]
         = -2 × [ln((1-p)^(N-x) × p^x) - ln((1-p_hat)^(N-x) × p_hat^x)]

Under H0, LR_POF ~ χ²(1) asymptotically.

INTERPRETATION:
===============
- If LR_POF is large (p-value < α), we REJECT H0.
  → The observed exception rate is statistically inconsistent with
    the assumed coverage probability at significance level α.
  → This is evidence that the VaR model may be miscalibrated.

- If LR_POF is small (p-value ≥ α), we FAIL TO REJECT H0.
  → We cannot statistically distinguish the observed from the expected
    exception rate at this significance level.
  → This does NOT "prove" the model is correct.

IMPORTANT DISCLAIMERS:
======================
1. Failing the Kupiec test does NOT prove the model is definitively wrong.
2. Passing the Kupiec test does NOT prove the model is correct.
3. With small samples (N < 250), the test has low statistical power.
4. The test only checks the quantity of exceptions, not their timing.

Reference: Kupiec, P. H. (1995). "Techniques for Verifying the Accuracy
of Risk Measurement Models." Journal of Derivatives, 3(2), 73-84.
"""

import logging
import warnings
from typing import Optional

import numpy as np
from scipy.stats import chi2

logger = logging.getLogger(__name__)


def kupiec_pof_test(
    n_observations: int,
    n_violations: int,
    confidence_level: float = 0.99,
    alpha: float = 0.05,
) -> dict:
    """
    Perform Kupiec Proportion of Failures (POF) test.

    Parameters
    ----------
    n_observations : int
        Total number of backtesting observations (N).
    n_violations : int
        Number of VaR exceptions observed (x).
    confidence_level : float
        VaR confidence level (e.g., 0.99). Theoretical violation prob = 1-CL.
    alpha : float
        Significance level for hypothesis test (e.g., 0.05).

    Returns
    -------
    dict containing:
        - n_observations: N
        - n_violations: x
        - expected_violation_rate: p (theoretical)
        - observed_violation_rate: p_hat = x/N
        - expected_violations: N × p
        - lr_statistic: LR_POF
        - p_value: chi-square p-value with 1 degree of freedom
        - reject_h0: True if we reject H0 (i.e., test FAILS)
        - test_result: "FAIL" if reject, "PASS" if fail-to-reject
        - interpretation: plain-language explanation
    """
    N = n_observations
    x = n_violations
    p = 1.0 - confidence_level  # theoretical violation probability

    if N <= 0:
        return _empty_result(p, alpha, reason="No observations")

    # Observed violation rate
    p_hat = x / N

    # --- Compute Log-Likelihood Ratio ---
    # We compute in log space to avoid numerical underflow with large N

    # Log-likelihood under H0: L(p)
    # ln((1-p)^(N-x) × p^x)  — handle edge cases x=0 and x=N
    ll_h0 = _log_likelihood(p, N, x)

    # Log-likelihood under H1: L(p_hat)
    # ln((1-p_hat)^(N-x) × p_hat^x)
    ll_h1 = _log_likelihood(p_hat, N, x)

    # LR_POF = -2 × (ll_h0 - ll_h1)
    if np.isnan(ll_h0) or np.isnan(ll_h1) or np.isinf(ll_h0) or np.isinf(ll_h1):
        lr_stat = np.nan
        p_value = np.nan
        logger.warning(
            f"Kupiec: log-likelihood is nan/inf. ll_h0={ll_h0:.4f}, ll_h1={ll_h1:.4f}"
            f" (N={N}, x={x}, p={p:.4f}, p_hat={p_hat:.4f})"
        )
    else:
        lr_stat = -2.0 * (ll_h0 - ll_h1)
        # LR_POF ~ chi-square with 1 degree of freedom under H0
        p_value = float(chi2.sf(lr_stat, df=1))  # sf = 1 - CDF = p-value

    # Reject H0 if p-value < significance level
    if np.isnan(p_value):
        reject_h0 = None
        test_result = "INCONCLUSIVE"
    else:
        reject_h0 = bool(p_value < alpha)
        test_result = "FAIL" if reject_h0 else "PASS"

    result = {
        "n_observations": N,
        "n_violations": x,
        "expected_violation_rate": p,
        "observed_violation_rate": float(p_hat),
        "expected_violations": float(N * p),
        "lr_statistic": float(lr_stat) if not np.isnan(lr_stat) else None,
        "p_value": float(p_value) if not np.isnan(p_value) else None,
        "reject_h0": reject_h0,
        "test_result": test_result,
        "significance_level": alpha,
        "interpretation": _interpretation(test_result, p_value, alpha, x, N, p, p_hat),
    }

    logger.info(
        f"Kupiec POF: N={N}, x={x}, p_hat={p_hat:.4f}, "
        f"LR={lr_stat:.3f}, p-value={p_value:.4f}, result={test_result}"
    )

    return result


def _log_likelihood(prob: float, N: int, x: int) -> float:
    """
    Compute log-likelihood: ln((1-prob)^(N-x) × prob^x).

    Handles edge cases:
    - prob = 0 with x > 0: -inf (impossible event)
    - prob = 1 with x < N: -inf (impossible event)
    - prob = 0 with x = 0: uses the log(1-prob)*(N) = 0
    - prob = 1 with x = N: uses log(prob)*x = 0

    Parameters
    ----------
    prob : float
        Violation probability.
    N : int
        Total observations.
    x : int
        Number of violations.

    Returns
    -------
    float
        Log-likelihood value.
    """
    if x == 0 and prob == 0.0:
        return 0.0  # ln(1^N × 0^0) = 0 (by convention 0^0 = 1)
    if x == N and prob == 1.0:
        return 0.0  # ln(0^0 × 1^N) = 0

    # Guard against log(0) with small epsilon
    eps = 1e-10
    prob = np.clip(prob, eps, 1.0 - eps)

    ll = (N - x) * np.log(1.0 - prob) + x * np.log(prob)
    return float(ll)


def _interpretation(
    test_result: str,
    p_value: Optional[float],
    alpha: float,
    x: int,
    N: int,
    p: float,
    p_hat: float,
) -> str:
    """Generate plain-language interpretation of the Kupiec test result."""
    if test_result == "INCONCLUSIVE":
        return "Test inconclusive due to edge-case inputs (e.g., zero violations)."

    expected = N * p
    direction = "fewer" if x < expected else "more"

    if test_result == "FAIL":
        return (
            f"REJECT H0 at α={alpha:.0%}: The observed violation rate ({p_hat:.2%}) is "
            f"statistically inconsistent with the expected rate ({p:.2%}). "
            f"Observed {x} violations vs. {expected:.1f} expected ({direction} than expected). "
            f"p-value = {p_value:.4f} < {alpha}. "
            "This suggests the VaR model may be over- or under-estimating risk. "
            "Note: this does not definitively prove the model is wrong."
        )
    else:
        return (
            f"FAIL TO REJECT H0 at α={alpha:.0%}: The observed violation rate ({p_hat:.2%}) "
            f"is statistically consistent with the expected rate ({p:.2%}). "
            f"Observed {x} violations vs. {expected:.1f} expected. "
            f"p-value = {p_value:.4f} ≥ {alpha}. "
            "This does NOT prove the model is correct — insufficient evidence to reject."
        )


def _empty_result(p: float, alpha: float, reason: str) -> dict:
    """Return empty result for edge cases."""
    return {
        "n_observations": 0,
        "n_violations": 0,
        "expected_violation_rate": p,
        "observed_violation_rate": None,
        "expected_violations": None,
        "lr_statistic": None,
        "p_value": None,
        "reject_h0": None,
        "test_result": "INCONCLUSIVE",
        "significance_level": alpha,
        "interpretation": reason,
    }
