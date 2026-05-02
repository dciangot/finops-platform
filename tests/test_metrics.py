"""Tests for finops_backtest.metrics – both pure-Python and C-backed paths."""

from __future__ import annotations

import math
import pytest

from finops_backtest.metrics import (
    anomaly_score,
    array_mean,
    array_stddev,
    array_variance,
    cost_efficiency,
    cumulative_savings,
    payback_period,
    roi,
    savings_rate,
    trend_slope,
    using_c_extension,
)


# ---------------------------------------------------------------------------
# savings_rate
# ---------------------------------------------------------------------------

class TestSavingsRate:
    def test_basic(self):
        assert savings_rate(100.0, 75.0) == pytest.approx(25.0)

    def test_no_savings(self):
        assert savings_rate(100.0, 100.0) == pytest.approx(0.0)

    def test_full_savings(self):
        assert savings_rate(100.0, 0.0) == pytest.approx(100.0)

    def test_negative_original(self):
        assert savings_rate(-10.0, 5.0) == pytest.approx(0.0)

    def test_zero_original(self):
        assert savings_rate(0.0, 5.0) == pytest.approx(0.0)

    def test_overspend(self):
        # Optimised cost is higher → negative savings rate
        assert savings_rate(100.0, 120.0) == pytest.approx(-20.0)


# ---------------------------------------------------------------------------
# cost_efficiency
# ---------------------------------------------------------------------------

class TestCostEfficiency:
    def test_better_than_baseline(self):
        # actual < baseline → ratio > 1
        assert cost_efficiency(80.0, 100.0) == pytest.approx(1.25)

    def test_equal(self):
        assert cost_efficiency(100.0, 100.0) == pytest.approx(1.0)

    def test_zero_actual(self):
        assert cost_efficiency(0.0, 100.0) == pytest.approx(0.0)

    def test_negative_actual(self):
        assert cost_efficiency(-5.0, 100.0) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# roi
# ---------------------------------------------------------------------------

class TestROI:
    def test_positive(self):
        assert roi(150.0, 100.0) == pytest.approx(50.0)

    def test_breakeven(self):
        assert roi(100.0, 100.0) == pytest.approx(0.0)

    def test_loss(self):
        assert roi(50.0, 100.0) == pytest.approx(-50.0)

    def test_zero_investment(self):
        assert roi(100.0, 0.0) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# payback_period
# ---------------------------------------------------------------------------

class TestPaybackPeriod:
    def test_basic(self):
        assert payback_period(1200.0, 100.0) == pytest.approx(12.0)

    def test_immediate(self):
        assert payback_period(0.0, 100.0) == pytest.approx(0.0)

    def test_zero_monthly_savings(self):
        assert payback_period(1000.0, 0.0) == pytest.approx(-1.0)

    def test_negative_monthly_savings(self):
        assert payback_period(1000.0, -50.0) == pytest.approx(-1.0)


# ---------------------------------------------------------------------------
# array_mean
# ---------------------------------------------------------------------------

class TestArrayMean:
    def test_uniform(self):
        assert array_mean([10.0, 10.0, 10.0]) == pytest.approx(10.0)

    def test_mixed(self):
        assert array_mean([10.0, 20.0, 30.0]) == pytest.approx(20.0)

    def test_single(self):
        assert array_mean([42.0]) == pytest.approx(42.0)

    def test_empty(self):
        assert array_mean([]) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# array_variance
# ---------------------------------------------------------------------------

class TestArrayVariance:
    def test_known(self):
        # Population variance of [2,4,4,4,5,5,7,9] = 4.0
        data = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
        assert array_variance(data) == pytest.approx(4.0)

    def test_constant(self):
        assert array_variance([5.0, 5.0, 5.0]) == pytest.approx(0.0)

    def test_empty(self):
        assert array_variance([]) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# array_stddev
# ---------------------------------------------------------------------------

class TestArrayStddev:
    def test_known(self):
        data = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
        assert array_stddev(data) == pytest.approx(2.0)

    def test_constant(self):
        assert array_stddev([3.0, 3.0, 3.0]) == pytest.approx(0.0)

    def test_empty(self):
        assert array_stddev([]) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# anomaly_score
# ---------------------------------------------------------------------------

class TestAnomalyScore:
    def test_exact_2sigma(self):
        assert anomaly_score(150.0, 100.0, 25.0) == pytest.approx(2.0)

    def test_no_deviation(self):
        assert anomaly_score(100.0, 100.0, 25.0) == pytest.approx(0.0)

    def test_negative(self):
        assert anomaly_score(50.0, 100.0, 25.0) == pytest.approx(-2.0)

    def test_zero_stddev(self):
        assert anomaly_score(200.0, 100.0, 0.0) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# trend_slope
# ---------------------------------------------------------------------------

class TestTrendSlope:
    def test_linear_increase(self):
        # 10, 20, 30 → slope = 10
        assert trend_slope([10.0, 20.0, 30.0]) == pytest.approx(10.0)

    def test_linear_decrease(self):
        assert trend_slope([30.0, 20.0, 10.0]) == pytest.approx(-10.0)

    def test_flat(self):
        assert trend_slope([5.0, 5.0, 5.0]) == pytest.approx(0.0)

    def test_single(self):
        assert trend_slope([100.0]) == pytest.approx(0.0)

    def test_empty(self):
        assert trend_slope([]) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# cumulative_savings
# ---------------------------------------------------------------------------

class TestCumulativeSavings:
    def test_basic(self):
        orig = [100.0, 120.0, 110.0]
        opt = [80.0, 90.0, 85.0]
        assert cumulative_savings(orig, opt) == pytest.approx(75.0)

    def test_no_savings(self):
        orig = [100.0, 100.0]
        opt = [100.0, 100.0]
        assert cumulative_savings(orig, opt) == pytest.approx(0.0)

    def test_negative_savings(self):
        orig = [80.0, 90.0]
        opt = [100.0, 110.0]
        assert cumulative_savings(orig, opt) == pytest.approx(-40.0)


# ---------------------------------------------------------------------------
# Extension status
# ---------------------------------------------------------------------------

def test_using_c_extension_returns_bool():
    result = using_c_extension()
    assert isinstance(result, bool)
