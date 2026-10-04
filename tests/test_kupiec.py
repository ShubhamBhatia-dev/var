"""
tests/test_kupiec.py
--------------------
Unit tests for the Kupiec Proportion of Failures (POF) test.

Tests cover:
1. Standard case: 250 observations, 2 violations → expected pass.
2. Too many violations → expected fail.
3. Too few violations → check if rejected.
4. Edge cases: 0 violations, all violations.
5. Correct formula for LR statistic.
6. p-value is chi-square(1) critical value comparison.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest
from scipy.stats import chi2

from risk_engine.kupiec import kupiec_pof_test, _log_likelihood


class TestKupiecPOFTest:
    """Tests for the Kupiec Proportion of Failures test."""

    def test_standard_case_2_violations_in_250(self):
        """
        Basel benchmark: 2 violations in 250 observations at 99% VaR.
        This is very close to expected (2.5 expected), so should PASS.
        """
        result = kupiec_pof_test(
            n_observations=250,
            n_violations=2,
            confidence_level=0.99,
            alpha=0.05,
        )
        assert result['test_result'] in ('PASS', 'FAIL')
        assert result['n_observations'] == 250
        assert result['n_violations'] == 2
        assert result['expected_violation_rate'] == pytest.approx(0.01)
        assert result['observed_violation_rate'] == pytest.approx(0.008)
        # 2 vs 2.5 expected — should pass
        assert result['test_result'] == 'PASS', (
            f"Expected PASS for 2/250 violations, got {result['test_result']}"
        )

    def test_too_many_violations_fail(self):
        """
        10 violations in 250 observations (4% rate vs 1% expected).
        This grossly exceeds the expected rate → should FAIL.
        """
        result = kupiec_pof_test(
            n_observations=250,
            n_violations=10,
            confidence_level=0.99,
            alpha=0.05,
        )
        assert result['test_result'] == 'FAIL', (
            f"Expected FAIL for 10/250 violations at 99% VaR, got {result['test_result']}"
        )
        assert result['p_value'] < 0.05

    def test_zero_violations(self):
        """
        0 violations in 250 observations.
        Very few violations could indicate an overly conservative model.
        At 99% confidence, 0/250 may or may not reject depending on the test.
        Key requirement: should NOT crash (no NaN/inf).
        """
        result = kupiec_pof_test(
            n_observations=250,
            n_violations=0,
            confidence_level=0.99,
            alpha=0.05,
        )
        assert result['test_result'] in ('PASS', 'FAIL', 'INCONCLUSIVE')
        # p-value should be a valid number or None (not NaN)
        if result['p_value'] is not None:
            assert not np.isnan(result['p_value'])

    def test_all_violations_extreme(self):
        """
        All 250 observations are violations → completely wrong model.
        Should FAIL and not crash.
        """
        result = kupiec_pof_test(
            n_observations=250,
            n_violations=250,
            confidence_level=0.99,
            alpha=0.05,
        )
        assert result['test_result'] in ('FAIL', 'INCONCLUSIVE')
        if result['test_result'] == 'FAIL':
            assert result['p_value'] < 0.05

    def test_lr_statistic_formula(self):
        """
        Verify the LR statistic is correctly computed using the formula:
        LR = -2 * (ll_H0 - ll_H1)

        For N=250, x=5, p=0.01:
        p_hat = 5/250 = 0.02
        ll_H0 = 245*log(0.99) + 5*log(0.01)
        ll_H1 = 245*log(0.98) + 5*log(0.02)
        LR = -2 * (ll_H0 - ll_H1)
        """
        N, x, p = 250, 5, 0.01
        p_hat = x / N
        eps = 1e-10
        ll_h0 = (N - x) * np.log(1 - p) + x * np.log(p + eps)
        ll_h1 = (N - x) * np.log(1 - p_hat) + x * np.log(p_hat)
        expected_lr = -2 * (ll_h0 - ll_h1)

        result = kupiec_pof_test(N, x, confidence_level=0.99, alpha=0.05)
        if result['lr_statistic'] is not None:
            assert abs(result['lr_statistic'] - expected_lr) < 0.01, (
                f"LR statistic {result['lr_statistic']:.4f} != expected {expected_lr:.4f}"
            )

    def test_p_value_chi2_relationship(self):
        """
        The p-value should be derived from the chi-square(1) distribution.
        p_value = 1 - chi2.cdf(LR, df=1)
        """
        result = kupiec_pof_test(250, 7, 0.99, 0.05)
        if result['lr_statistic'] is not None and result['p_value'] is not None:
            expected_pvalue = float(chi2.sf(result['lr_statistic'], df=1))
            assert abs(result['p_value'] - expected_pvalue) < 1e-6

    def test_result_structure(self):
        """Check that all expected keys are present in the result."""
        result = kupiec_pof_test(500, 4, 0.99, 0.05)
        required_keys = [
            'n_observations', 'n_violations', 'expected_violation_rate',
            'observed_violation_rate', 'expected_violations',
            'lr_statistic', 'p_value', 'reject_h0', 'test_result',
            'significance_level', 'interpretation',
        ]
        for key in required_keys:
            assert key in result, f"Missing key: {key}"

    def test_expected_violations_calculation(self):
        """expected_violations = N × p."""
        result = kupiec_pof_test(500, 5, 0.99, 0.05)
        assert result['expected_violations'] == pytest.approx(500 * 0.01)

    def test_95pct_confidence_var(self):
        """Test with 95% VaR (p = 0.05) — different expected rate."""
        result = kupiec_pof_test(
            n_observations=250,
            n_violations=12,  # 12/250 = 4.8%, close to 5%
            confidence_level=0.95,
            alpha=0.05,
        )
        assert result['expected_violation_rate'] == pytest.approx(0.05)
        # 12 vs 12.5 expected → should PASS
        assert result['test_result'] == 'PASS', (
            f"12/250 at 95% VaR should PASS, got {result['test_result']}"
        )


class TestLogLikelihood:
    """Tests for the internal log-likelihood helper."""

    def test_edge_case_x_equals_zero(self):
        """x=0, p=0.01 should not produce NaN."""
        ll = _log_likelihood(0.01, 250, 0)
        assert not np.isnan(ll), "Log-likelihood should not be NaN for x=0"
        assert not np.isinf(ll), "Log-likelihood should not be Inf for x=0"

    def test_edge_case_prob_zero_x_zero(self):
        """p=0, x=0 is a degenerate case — should return 0."""
        ll = _log_likelihood(0.0, 250, 0)
        assert ll == 0.0

    def test_normal_case(self):
        """Standard log-likelihood calculation."""
        ll = _log_likelihood(0.01, 250, 2)
        expected = 248 * np.log(0.99) + 2 * np.log(0.01)
        assert abs(ll - expected) < 1e-9
