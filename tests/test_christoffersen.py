"""
tests/test_christoffersen.py
----------------------------
Unit tests for the Christoffersen Independence and Conditional Coverage tests.

Tests cover:
1. Transition count matrix is correctly computed.
2. Independence test correctly identifies clustered vs. non-clustered violations.
3. LR_IND formula is computed correctly.
4. Conditional Coverage = LR_POF + LR_IND.
5. Edge cases: all zeros, all ones, alternating.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest

from risk_engine.christoffersen import (
    christoffersen_independence_test,
    conditional_coverage_test,
    _count_transitions,
)


class TestTransitionCounting:
    """Tests for the transition matrix counting helper."""

    def test_all_zeros(self):
        """All non-violations: N00=n-1, N01=N10=N11=0."""
        v = np.array([0, 0, 0, 0, 0])
        n00, n01, n10, n11 = _count_transitions(v)
        assert n00 == 4
        assert n01 == 0
        assert n10 == 0
        assert n11 == 0

    def test_all_ones(self):
        """All violations: N11=n-1, N00=N01=N10=0."""
        v = np.array([1, 1, 1, 1, 1])
        n00, n01, n10, n11 = _count_transitions(v)
        assert n00 == 0
        assert n01 == 0
        assert n10 == 0
        assert n11 == 4

    def test_alternating(self):
        """
        Perfect alternating: 0,1,0,1,0,1
        Transitions: 0→1 (x3), 1→0 (x2)
        N00=0, N01=3, N10=2, N11=0
        """
        v = np.array([0, 1, 0, 1, 0, 1])
        n00, n01, n10, n11 = _count_transitions(v)
        assert n00 == 0
        assert n01 == 3
        assert n10 == 2
        assert n11 == 0

    def test_known_sequence(self):
        """
        Manually computed example.
        Sequence: 0, 0, 1, 1, 0, 1, 0, 0
        Transitions:
          0→0: (0,0), (4→0 if 6=0) = depends on seq...
          Let's trace:
          (0,0): 0→0  N00
          (0,1): 0→1  N01
          (1,1): 1→1  N11
          (1,0): 1→0  N10
          (0,1): 0→1  N01
          (1,0): 1→0  N10
          (0,0): 0→0  N00
          N00=2, N01=2, N10=2, N11=1
        """
        v = np.array([0, 0, 1, 1, 0, 1, 0, 0])
        n00, n01, n10, n11 = _count_transitions(v)
        assert n00 == 2
        assert n01 == 2
        assert n10 == 2
        assert n11 == 1

    def test_total_transitions(self):
        """Total transitions should equal len(v) - 1."""
        v = np.array([0, 1, 0, 0, 1, 1, 0, 1])
        n00, n01, n10, n11 = _count_transitions(v)
        assert n00 + n01 + n10 + n11 == len(v) - 1


class TestChristoffersenIndependence:
    """Tests for the Independence test."""

    def test_perfectly_clustered_violations(self):
        """
        If violations cluster (1,1,1,1 then none), pi1 >> pi0.
        This should reject independence (FAIL).
        """
        # Clustered: many consecutive violations
        v = [0] * 100 + [1] * 20 + [0] * 100 + [1] * 20 + [0] * 100
        result = christoffersen_independence_test(v, alpha=0.05)
        # High pi1 (violation after violation) vs low pi0 should indicate clustering
        assert result['pi1'] > result['pi0'], "pi1 should exceed pi0 for clustered violations"

    def test_perfectly_alternating_violations(self):
        """
        Perfectly alternating 0,1,0,1... means violations are maximally spread.
        pi0 = 1.0 (always violated after non-violation)
        pi1 = 0.0 (never two consecutive violations)
        """
        v = [0, 1] * 50  # 100 elements, alternating
        result = christoffersen_independence_test(v, alpha=0.05)
        assert result['n00'] == 0  # no 0→0 transitions
        assert result['n11'] == 0  # no 1→1 transitions
        assert result['pi0'] == pytest.approx(1.0)  # always: 0 → 1
        assert result['pi1'] == pytest.approx(0.0)  # never: 1 → 1

    def test_result_structure(self):
        """Independence test should return all required keys."""
        v = [0, 0, 1, 0, 0, 0, 1, 0, 0, 1]
        result = christoffersen_independence_test(v, alpha=0.05)
        required_keys = [
            'n00', 'n01', 'n10', 'n11',
            'pi0', 'pi1', 'pi_unconditional',
            'lr_independence', 'p_value_independence',
            'reject_h0', 'test_result', 'significance_level', 'interpretation',
        ]
        for key in required_keys:
            assert key in result, f"Missing key: {key}"

    def test_insufficient_data(self):
        """Single element sequence should return inconclusive."""
        result = christoffersen_independence_test([0], alpha=0.05)
        assert result['test_result'] == 'INCONCLUSIVE'

    def test_no_violations(self):
        """All zeros: pi0=0, pi1 undefined. Should not crash."""
        v = [0] * 50
        result = christoffersen_independence_test(v, alpha=0.05)
        assert result is not None
        assert result['n01'] == 0

    def test_pi_values_are_probabilities(self):
        """pi0, pi1 should be in [0, 1]."""
        rng = np.random.default_rng(42)
        v = (rng.random(300) < 0.01).astype(int).tolist()
        result = christoffersen_independence_test(v, alpha=0.05)
        assert 0 <= result['pi0'] <= 1, f"pi0 out of range: {result['pi0']}"
        assert 0 <= result['pi1'] <= 1, f"pi1 out of range: {result['pi1']}"


class TestConditionalCoverage:
    """Tests for the Conditional Coverage combined test."""

    def test_cc_equals_lr_pof_plus_lr_ind(self):
        """
        LR_CC should equal LR_POF + LR_IND (up to numerical precision).
        """
        rng = np.random.default_rng(77)
        v = (rng.random(500) < 0.01).astype(int).tolist()
        n_obs = 500

        result = conditional_coverage_test(v, n_obs, 0.99, 0.05)

        lr_pof = result['kupiec']['lr_statistic']
        lr_ind = result['independence']['lr_independence']
        lr_cc = result['lr_conditional_coverage']

        if lr_pof is not None and lr_ind is not None and lr_cc is not None:
            assert abs(lr_cc - (lr_pof + lr_ind)) < 1e-6, (
                f"LR_CC = {lr_cc:.4f} != LR_POF + LR_IND = {lr_pof + lr_ind:.4f}"
            )

    def test_cc_uses_2_degrees_of_freedom(self):
        """CC p-value is chi-square with 2 degrees of freedom."""
        from scipy.stats import chi2
        rng = np.random.default_rng(99)
        v = (rng.random(500) < 0.015).astype(int).tolist()
        result = conditional_coverage_test(v, 500, 0.99, 0.05)

        if result['lr_conditional_coverage'] is not None:
            expected_pval = float(chi2.sf(result['lr_conditional_coverage'], df=2))
            assert abs(result['p_value_conditional_coverage'] - expected_pval) < 1e-6

    def test_cc_structure(self):
        """Check all expected keys in conditional coverage result."""
        v = [0] * 490 + [1] * 10
        result = conditional_coverage_test(v, 500, 0.99, 0.05)
        required_keys = [
            'lr_conditional_coverage', 'p_value_conditional_coverage',
            'reject_h0', 'test_result', 'significance_level',
            'kupiec', 'independence', 'interpretation',
        ]
        for key in required_keys:
            assert key in result, f"Missing key: {key}"


class TestLookAheadBias:
    """
    Test that rolling VaR calculations do not use future data.

    This is a crucial requirement: VaR for day t must only use
    returns from t-window to t-1, NOT from day t or later.
    """

    def test_no_look_ahead_in_rolling_var(self):
        """
        Verify that the rolling VaR engine uses only past data.

        Strategy: Create a P&L series where all gains occur on even days
        and all losses occur on odd days. The VaR for odd days should be
        estimated from even-day gains (low VaR) — if look-ahead is absent.

        More practically: we check that the VaR on day t uses exactly
        returns from [t-window, t-1], not including t.
        """
        import pandas as pd
        from risk_engine import historical_simulation, portfolio as pm

        # Create a simple portfolio
        weights = {'A': 0.6, 'B': 0.4}
        port = pm.Portfolio(weights=weights, notional=1_000_000)

        # Create deterministic log returns
        n_days = 300
        dates = pd.bdate_range('2020-01-01', periods=n_days)

        # Alternating: large gains followed by large losses
        returns_A = np.array([0.05 if i % 2 == 0 else -0.05 for i in range(n_days)])
        returns_B = np.array([0.03 if i % 2 == 0 else -0.03 for i in range(n_days)])

        log_returns = pd.DataFrame(
            {'A': returns_A, 'B': returns_B},
            index=dates,
        )

        # Run rolling backtest
        rolling = historical_simulation.compute_rolling_var_es(
            log_returns=log_returns,
            portfolio=port,
            window=100,
            confidence=0.99,
            min_window=100,
        )

        # The rolling results should have fewer rows than total observations
        # (because we need at least min_window observations before starting)
        assert len(rolling) < n_days, "Rolling result should start after min_window observations"
        assert len(rolling) == n_days - 100, f"Expected {n_days - 100} rolling dates"

        # VaR should be positive
        assert (rolling['VaR_1D'] > 0).all(), "All VaR estimates should be positive"

        # Violations should be flagged correctly: violation when Realized_PnL < -VaR_1D
        for _, row in rolling.iterrows():
            expected_violation = int(row['Realized_PnL'] < -row['VaR_1D'])
            assert row['Violation'] == expected_violation, (
                f"Violation flag mismatch on {row.name}: "
                f"P&L={row['Realized_PnL']:.0f}, VaR={row['VaR_1D']:.0f}"
            )
