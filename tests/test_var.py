"""
tests/test_var.py
-----------------
Unit tests for the VaR calculation module.

Tests verify that:
1. VaR equals the negative of the empirical (1-confidence) quantile.
2. VaR is always non-negative.
3. VaR of an all-positive P&L distribution is 0.
4. VaR is invariant to ordering of the P&L array.
5. The sqrt-of-time approximation is correctly computed.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest

from risk_engine.var import calculate_var, calculate_var_sqrt_of_time


class TestCalculateVar:
    """Tests for the 1-day VaR calculation."""

    def test_known_distribution_99pct(self):
        """
        Given a known P&L distribution, VaR should equal -quantile(P&L, 0.01).

        Example: 100 uniformly distributed P&L values from -1 to 1.
        The 1st percentile of uniform(-1, 1) is approximately -0.98.
        VaR = 0.98 (positive loss amount).
        """
        np.random.seed(42)
        pnl = np.linspace(-1.0, 1.0, 100)  # perfectly uniform -1 to 1
        confidence = 0.99
        expected_var = -np.quantile(pnl, 0.01)
        calculated_var = calculate_var(pnl, confidence)
        assert abs(calculated_var - expected_var) < 1e-9, (
            f"VaR {calculated_var:.6f} != expected {expected_var:.6f}"
        )

    def test_var_is_positive(self):
        """VaR should always be non-negative."""
        pnl = np.array([-500, -200, -100, 50, 200, 300, 400, 500])
        var = calculate_var(pnl, confidence=0.99)
        assert var >= 0, f"VaR should be non-negative, got {var}"

    def test_var_all_positive_pnl(self):
        """If all P&L values are positive, VaR should be 0 (no losses)."""
        pnl = np.array([100.0, 200.0, 300.0, 400.0, 500.0])
        var = calculate_var(pnl, confidence=0.99)
        assert var == 0.0, f"VaR of all-positive P&L should be 0, got {var}"

    def test_var_ordering_invariance(self):
        """VaR should be the same regardless of array ordering."""
        pnl = np.array([-400, 100, -200, 300, -100, 500, -50, 250])
        var_original = calculate_var(pnl, confidence=0.99)
        var_sorted = calculate_var(np.sort(pnl), confidence=0.99)
        var_reversed = calculate_var(pnl[::-1], confidence=0.99)
        assert abs(var_original - var_sorted) < 1e-9
        assert abs(var_original - var_reversed) < 1e-9

    def test_var_convention_positive_loss(self):
        """
        VaR represents a positive loss amount.

        If 1st percentile of P&L = -$250,000, then VaR = $250,000.
        """
        # Construct array where 1st percentile is robustly -250,000
        # With 1000 elements, 1% is 10 elements. If worst 15 are -250,000,
        # the 1st percentile will fall within those -250,000 elements.
        pnl = np.array([-250_000] * 15 + [100_000] * 985)
        var = calculate_var(pnl, confidence=0.99)
        assert var > 0, "VaR should be positive (loss amount)"
        assert abs(var - 250_000) < 1.0, f"Expected VaR ≈ 250,000, got {var:.2f}"

    def test_var_empty_array(self):
        """Empty array should return 0 without crashing."""
        var = calculate_var(np.array([]), confidence=0.99)
        assert var == 0.0

    def test_var_single_value(self):
        """Single P&L value: VaR = max(0, -pnl)."""
        var_loss = calculate_var(np.array([-500.0]), confidence=0.99)
        assert var_loss == 500.0
        var_gain = calculate_var(np.array([500.0]), confidence=0.99)
        assert var_gain == 0.0

    def test_var_confidence_levels(self):
        """Higher confidence level should give higher or equal VaR."""
        pnl = np.random.default_rng(123).normal(0, 100, 1000)
        var_95 = calculate_var(pnl, confidence=0.95)
        var_99 = calculate_var(pnl, confidence=0.99)
        var_99_9 = calculate_var(pnl, confidence=0.999)
        assert var_95 <= var_99, f"99% VaR should >= 95% VaR: {var_99:.2f} < {var_95:.2f}"
        assert var_99 <= var_99_9, f"99.9% VaR should >= 99% VaR"

    def test_var_large_portfolio(self):
        """Test with a realistic portfolio P&L distribution."""
        # Simulate 500 days of P&L for a $10M portfolio
        rng = np.random.default_rng(42)
        pnl = rng.normal(5000, 150_000, 500)  # small positive drift, large vol
        var = calculate_var(pnl, confidence=0.99)
        expected = -np.quantile(pnl, 0.01)
        assert abs(var - expected) < 1e-6


class TestSqrtOfTimeApproximation:
    """Tests for the square-root-of-time VaR approximation."""

    def test_sqrt_10_scaling(self):
        """sqrt(10) scaling should multiply 1-day VaR by sqrt(10)."""
        var_1d = 100_000.0
        var_10d_approx = calculate_var_sqrt_of_time(var_1d, holding_period=10)
        expected = var_1d * np.sqrt(10)
        assert abs(var_10d_approx - expected) < 1e-6, (
            f"Expected {expected:.2f}, got {var_10d_approx:.2f}"
        )

    def test_sqrt_1_returns_same(self):
        """sqrt(1) should return the same VaR."""
        var_1d = 200_000.0
        var_1d_approx = calculate_var_sqrt_of_time(var_1d, holding_period=1)
        assert abs(var_1d_approx - var_1d) < 1e-6

    def test_always_larger_than_1d(self):
        """Multi-day approximation should always exceed 1-day VaR (for h > 1)."""
        var_1d = 50_000.0
        for h in [2, 5, 10, 20]:
            var_hd = calculate_var_sqrt_of_time(var_1d, holding_period=h)
            assert var_hd > var_1d, f"VaR_{h}d should exceed VaR_1d"
