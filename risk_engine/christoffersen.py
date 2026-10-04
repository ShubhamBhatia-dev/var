"""
risk_engine/christoffersen.py
------------------------------
Christoffersen Independence Test and Conditional Coverage Test.

BACKGROUND:
===========
The Christoffersen (1998) test builds upon the Kupiec test by also
checking the INDEPENDENCE of VaR exceptions.

KEY INSIGHT:
============
Even if the TOTAL NUMBER of exceptions is correct (Kupiec PASS),
the exceptions might cluster together — occurring in consecutive days
during stressed periods. This is problematic because:

1. It means the risk model fails during exactly the periods when
   risk management is most critical (market crises).

2. It violates the statistical assumption that each day's VaR should
   independently have a 1% probability of being exceeded.

KUPIEC vs. CHRISTOFFERSEN:
==========================
Kupiec asks:    "Are there approximately the right NUMBER of exceptions?"
Christoffersen: "Are exceptions INDEPENDENT (non-clustered)?"

A model can PASS Kupiec but FAIL Christoffersen if exceptions cluster.
Conversely, a model can FAIL Kupiec (wrong count) but PASS Christoffersen
(non-clustered exceptions).

TRANSITION MATRIX:
==================
Define I_t = 1 if VaR is violated on day t, 0 otherwise.
Count transitions between states:

  N00 = count(I_{t-1}=0, I_t=0) — non-violation followed by non-violation
  N01 = count(I_{t-1}=0, I_t=1) — non-violation followed by violation
  N10 = count(I_{t-1}=1, I_t=0) — violation followed by non-violation
  N11 = count(I_{t-1}=1, I_t=1) — violation followed by violation

CONDITIONAL PROBABILITIES:
===========================
Under H1 (independence), each state is equally likely regardless of history:
  pi  = (N01 + N11) / (N00 + N01 + N10 + N11)

Under H0 (dependence allowed):
  pi0 = N01 / (N00 + N01)  — prob of violation given no violation yesterday
  pi1 = N11 / (N10 + N11)  — prob of violation given violation yesterday

INDEPENDENCE LIKELIHOOD RATIO:
================================
LR_IND = -2 × [ln L(pi) - ln L(pi0, pi1)]

Under H0 (independence), LR_IND ~ chi-square(1).

CONDITIONAL COVERAGE:
======================
Christoffersen (1998) also proposes a joint test:
  LR_CC = LR_POF + LR_IND ~ chi-square(2)

This tests BOTH that the fraction of exceptions is correct AND
that they are temporally independent.

Reference: Christoffersen, P. F. (1998). "Evaluating Interval Forecasts."
International Economic Review, 39(4), 841-862.
"""

import logging
from typing import Sequence, Tuple

import numpy as np
from scipy.stats import chi2

logger = logging.getLogger(__name__)


def christoffersen_independence_test(
    violations: Sequence[int],
    alpha: float = 0.05,
) -> dict:
    """
    Perform the Christoffersen Independence Test.

    Parameters
    ----------
    violations : sequence of int
        Binary sequence: 1 = VaR violation on that day, 0 = no violation.
        Must be in chronological order.
    alpha : float
        Significance level (default: 0.05).

    Returns
    -------
    dict containing:
        - n00, n01, n10, n11: transition counts
        - pi0: conditional prob of violation given no prior violation
        - pi1: conditional prob of violation given prior violation
        - pi: unconditional violation rate
        - lr_independence: LR_IND statistic
        - p_value_independence: chi-square p-value (1 df)
        - reject_h0: True if we reject independence hypothesis
        - test_result: "FAIL" (reject) or "PASS" (fail-to-reject)
        - interpretation: plain-language explanation
    """
    v = np.array(violations, dtype=int)

    if len(v) < 2:
        return _empty_independence_result(alpha, "Insufficient observations for independence test")

    # Count transitions
    n00, n01, n10, n11 = _count_transitions(v)

    # Conditional probabilities
    # pi0 = probability of violation given previous state = no violation
    pi0 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0.0
    # pi1 = probability of violation given previous state = violation
    pi1 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0.0
    # pi = unconditional violation probability from transitions
    total_trans = n00 + n01 + n10 + n11
    pi = (n01 + n11) / total_trans if total_trans > 0 else 0.0

    # Log-likelihoods
    # Under independence (H0 for independence test):
    #   L(pi) = pi^(N01+N11) × (1-pi)^(N00+N10)
    ll_independence = _log_likelihood_independence(pi, n00, n01, n10, n11)

    # Under the alternative (conditional on previous state):
    #   L(pi0, pi1) = pi0^N01 × (1-pi0)^N00 × pi1^N11 × (1-pi1)^N10
    ll_markov = _log_likelihood_markov(pi0, pi1, n00, n01, n10, n11)

    # LR_IND = -2 × (ll_independence - ll_markov)
    if np.isnan(ll_independence) or np.isnan(ll_markov):
        lr_ind = np.nan
        p_value = np.nan
    else:
        lr_ind = -2.0 * (ll_independence - ll_markov)
        p_value = float(chi2.sf(lr_ind, df=1))

    reject_h0 = bool(p_value < alpha) if not np.isnan(p_value) else None
    test_result = "FAIL" if reject_h0 else "PASS" if reject_h0 is not None else "INCONCLUSIVE"

    result = {
        "n00": int(n00),
        "n01": int(n01),
        "n10": int(n10),
        "n11": int(n11),
        "pi0": float(pi0),
        "pi1": float(pi1),
        "pi_unconditional": float(pi),
        "lr_independence": float(lr_ind) if not np.isnan(lr_ind) else None,
        "p_value_independence": float(p_value) if not np.isnan(p_value) else None,
        "reject_h0": reject_h0,
        "test_result": test_result,
        "significance_level": alpha,
        "interpretation": _independence_interpretation(test_result, p_value, alpha, pi0, pi1, pi),
    }

    logger.info(
        f"Christoffersen Independence: N00={n00}, N01={n01}, N10={n10}, N11={n11}, "
        f"pi0={pi0:.4f}, pi1={pi1:.4f}, LR_IND={lr_ind:.3f}, "
        f"p-value={p_value:.4f}, result={test_result}"
    )

    return result


def conditional_coverage_test(
    violations: Sequence[int],
    n_observations: int,
    confidence_level: float = 0.99,
    alpha: float = 0.05,
) -> dict:
    """
    Christoffersen Conditional Coverage Test.

    Combines Kupiec POF (proportion test) and Independence test.

    LR_CC = LR_POF + LR_IND ~ chi-square(2) under H0

    Tests jointly:
    1. Is the fraction of exceptions correct? (Kupiec)
    2. Are exceptions temporally independent? (Independence)

    Parameters
    ----------
    violations : sequence of int
        Binary violation sequence.
    n_observations : int
        Total number of backtesting observations.
    confidence_level : float
        VaR confidence level.
    alpha : float
        Significance level.

    Returns
    -------
    dict
        Combined conditional coverage test result.
    """
    from .kupiec import kupiec_pof_test

    n_violations = int(sum(violations))

    kupiec_result = kupiec_pof_test(
        n_observations, n_violations, confidence_level, alpha
    )
    independence_result = christoffersen_independence_test(violations, alpha)

    lr_pof = kupiec_result.get("lr_statistic")
    lr_ind = independence_result.get("lr_independence")

    if lr_pof is not None and lr_ind is not None:
        lr_cc = lr_pof + lr_ind
        p_value_cc = float(chi2.sf(lr_cc, df=2))  # 2 degrees of freedom
        reject_cc = bool(p_value_cc < alpha)
    else:
        lr_cc = None
        p_value_cc = None
        reject_cc = None

    test_result_cc = (
        "FAIL" if reject_cc else "PASS" if reject_cc is not None else "INCONCLUSIVE"
    )

    result = {
        "lr_conditional_coverage": float(lr_cc) if lr_cc is not None else None,
        "p_value_conditional_coverage": float(p_value_cc) if p_value_cc is not None else None,
        "reject_h0": reject_cc,
        "test_result": test_result_cc,
        "significance_level": alpha,
        "kupiec": kupiec_result,
        "independence": independence_result,
        "interpretation": (
            f"Conditional Coverage (Christoffersen): Tests both the frequency and "
            f"independence of VaR violations. LR_CC = LR_POF + LR_IND = "
            f"{lr_pof:.3f} + {lr_ind:.3f} = {lr_cc:.3f}. "
            f"p-value = {p_value_cc:.4f}. Result: {test_result_cc} "
            f"at α = {alpha:.0%} (chi-square with 2 d.f.)."
        ) if lr_cc is not None else "Conditional coverage could not be computed.",
    }

    return result


def _count_transitions(v: np.ndarray) -> Tuple[int, int, int, int]:
    """
    Count state transitions in a binary violation sequence.

    Parameters
    ----------
    v : np.ndarray
        Binary sequence (0 or 1).

    Returns
    -------
    (n00, n01, n10, n11)
    """
    prev = v[:-1]
    curr = v[1:]

    n00 = int(np.sum((prev == 0) & (curr == 0)))  # 0→0
    n01 = int(np.sum((prev == 0) & (curr == 1)))  # 0→1
    n10 = int(np.sum((prev == 1) & (curr == 0)))  # 1→0
    n11 = int(np.sum((prev == 1) & (curr == 1)))  # 1→1

    return n00, n01, n10, n11


def _log_likelihood_independence(
    pi: float, n00: int, n01: int, n10: int, n11: int
) -> float:
    """Log-likelihood under independence: L(pi) = pi^(N01+N11) × (1-pi)^(N00+N10)."""
    eps = 1e-10
    pi = np.clip(pi, eps, 1 - eps)
    return (n01 + n11) * np.log(pi) + (n00 + n10) * np.log(1 - pi)


def _log_likelihood_markov(
    pi0: float, pi1: float, n00: int, n01: int, n10: int, n11: int
) -> float:
    """
    Log-likelihood under Markov alternative:
    L(pi0, pi1) = pi0^N01 × (1-pi0)^N00 × pi1^N11 × (1-pi1)^N10
    """
    eps = 1e-10
    pi0 = np.clip(pi0, eps, 1 - eps)
    pi1 = np.clip(pi1, eps, 1 - eps)
    return (
        n01 * np.log(pi0)
        + n00 * np.log(1 - pi0)
        + n11 * np.log(pi1)
        + n10 * np.log(1 - pi1)
    )


def _independence_interpretation(
    test_result: str,
    p_value: float,
    alpha: float,
    pi0: float,
    pi1: float,
    pi: float,
) -> str:
    """Generate plain-language interpretation of the independence test."""
    if test_result == "INCONCLUSIVE":
        return "Independence test inconclusive due to insufficient data."

    clustering = "YES" if pi1 > pi0 else "NO"

    if test_result == "FAIL":
        return (
            f"REJECT H0 (independence) at α={alpha:.0%}: VaR violations are NOT independent. "
            f"Conditional violation prob given prior violation: {pi1:.2%} vs. "
            f"given no prior violation: {pi0:.2%}. "
            f"Evidence of violation clustering: {clustering}. "
            f"p-value = {p_value:.4f} < {alpha}."
        )
    else:
        return (
            f"FAIL TO REJECT H0 (independence) at α={alpha:.0%}: "
            f"Cannot statistically distinguish violations from an independent process. "
            f"pi0={pi0:.2%}, pi1={pi1:.2%}, unconditional={pi:.2%}. "
            f"p-value = {p_value:.4f} ≥ {alpha}."
        )


def _empty_independence_result(alpha: float, reason: str) -> dict:
    """Return empty result for edge cases."""
    return {
        "n00": 0, "n01": 0, "n10": 0, "n11": 0,
        "pi0": None, "pi1": None, "pi_unconditional": None,
        "lr_independence": None, "p_value_independence": None,
        "reject_h0": None, "test_result": "INCONCLUSIVE",
        "significance_level": alpha, "interpretation": reason,
    }
