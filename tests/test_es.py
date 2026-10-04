"""
tests/test_es.py
----------------
Unit tests for the Expected Shortfall calculation module.

Tests verify that:
1. ES = average of P&L values at or below the VaR threshold.
2. ES >= VaR always (ES is never less than VaR).
3. ES correctly captures the tail average.
4. ES handles edge cases gracefully.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest

from risk_engine.expected_shortfall import calculate_es, es_decomposition
from risk_engine.var import calculate_var


class TestCalculateES:
    """Tests for the Expected Shortfall calculation."""

    def test_es_is_average_of_tail(self):
        """
        ES should be the average of P&L values at or below the VaR threshold.

        With 100 observations at 99% confidence:
        - VaR threshold = 1st percentile
        - ES = average of all observations <= that threshold
        """
        pnl = np.linspace(-1.0, 1.0, 100)  # uniform -1 to 1
        confidence = 0.99
        var_threshold = np.quantile(pnl, 0.01)
        expected_es = -np.mean(pnl[pnl <= var_threshold])
        calculated_es = calculate_es(pnl, confidence)
        assert abs(calculated_es - expected_es) < 1e-9, (
            f"ES {calculated_es:.6f} != expected {expected_es:.6f}"
        )

    def test_es_greater_than_or_equal_var(self):
        """
        ES should always be >= VaR.
        (The tail average should be >= the tail threshold.)
        """
        rng = np.random.default_rng(42)
        for _ in range(20):
            pnl = rng.normal(0, 100_000, 500)
            var = calculate_var(pnl, confidence=0.99)
            es = calculate_es(pnl, confidence=0.99)
            assert es >= var - 1.0, (
                f"ES ({es:.2f}) should be >= VaR ({var:.2f})"
            )

    def test_es_larger_than_var_with_fat_tails(self):
        """
        For fat-tailed distributions (Student t), ES should be substantially
        larger than VaR, reflecting the heavy tail.
        """
        from scipy.stats import t as t_dist
        rng = np.random.default_rng(99)
        pnl = t_dist.rvs(df=3, scale=100_000, size=2000, random_state=rng)
        var = calculate_var(pnl, confidence=0.99)
        es = calculate_es(pnl, confidence=0.99)
        # For df=3, ES/VaR ratio should be > 1 (ES > VaR)
        assert es > var, f"For fat-tailed distribution, ES ({es:.0f}) should exceed VaR ({var:.0f})"

    def test_es_known_tail(self):
        """
        Explicit test with known tail values.

        P&L = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4]
        At 99% confidence, 1st percentile = -4.91 (approx -5 for this distribution)
        Tail observations at or below threshold = [-5]
        ES = 5.0
        """
        pnl = np.array([-5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0])
        confidence = 0.90  # 10 obs, 10th percentile = -4.1 approx
        var_threshold = np.quantile(pnl, 0.10)
        tail = pnl[pnl <= var_threshold]
        expected_es = -np.mean(tail)
        calculated_es = calculate_es(pnl, confidence=0.90)
        assert abs(calculated_es - expected_es) < 1e-6

    def test_es_positive_value(self):
        """ES should always be expressed as a positive loss amount."""
        pnl = np.array([-300, -200, -100, 50, 100, 200, 300, 400, 500, 600])
        es = calculate_es(pnl, confidence=0.99)
        assert es >= 0, f"ES should be non-negative, got {es}"

    def test_es_empty_array(self):
        """Empty array should return 0 without crashing."""
        es = calculate_es(np.array([]), confidence=0.99)
        assert es == 0.0

    def test_es_single_observation(self):
        """Single observation: ES = max(0, -pnl)."""
        es_loss = calculate_es(np.array([-500.0]), confidence=0.99)
        assert es_loss == 500.0

    def test_es_decomposition_structure(self):
        """es_decomposition should return expected keys."""
        pnl = np.random.default_rng(7).normal(0, 100_000, 500)
        result = es_decomposition(pnl, confidence=0.99)
        assert 'var' in result
        assert 'es' in result
        assert 'es_to_var_ratio' in result
        assert 'n_tail_observations' in result
        assert result['es'] >= result['var']

    def test_es_vs_var_interpretability(self):
        """
        VaR tells us the minimum loss in the worst 1%.
        ES tells us the AVERAGE loss in the worst 1%.
        ES >= VaR always.
        """
        rng = np.random.default_rng(55)
        pnl = rng.normal(-5_000, 200_000, 1000)

        var = calculate_var(pnl, 0.99)
        es = calculate_es(pnl, 0.99)

        # VaR = boundary; ES = average beyond boundary
        assert es >= var, "ES must be >= VaR: ES captures the tail average, not just the threshold"

        # ES/VaR ratio should be > 1 (usually 1.1 to 1.5 for near-normal)
        ratio = es / var if var > 0 else float('inf')
        assert ratio >= 1.0, f"ES/VaR ratio should be >= 1, got {ratio:.3f}"
