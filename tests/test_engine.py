"""Tests for the FinOps BacktestEngine."""

from __future__ import annotations

from datetime import date

import pytest

from finops_backtest import BacktestEngine, CostData, CostEntry, Strategy


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_entries(n: int = 12, base_cost: float = 1000.0, step: float = 50.0):
    """Create *n* monthly cost entries with a linear cost trend."""
    assert n <= 12, "Helper only supports up to 12 entries (one per month of 2024)"
    return [
        CostEntry(
            date=date(2024, i + 1, 1),
            service="EC2",
            cost=base_cost + i * step,
            usage=20.0,
            region="us-east-1",
        )
        for i in range(n)
    ]


@pytest.fixture
def cost_data():
    return CostData(entries=_make_entries(12))


@pytest.fixture
def flat_cost_data():
    entries = [
        CostEntry(date(2024, i + 1, 1), "S3", 500.0, 100.0, "eu-west-1")
        for i in range(12)
    ]
    return CostData(entries=entries)


@pytest.fixture
def reserved_strategy():
    return Strategy(
        name="Reserved Instances",
        description="30% discount",
        apply=lambda e: e.cost * 0.70,
    )


@pytest.fixture
def spot_strategy():
    return Strategy(
        name="Spot Instances",
        description="60% discount",
        apply=lambda e: e.cost * 0.40,
    )


# ---------------------------------------------------------------------------
# Basic run tests
# ---------------------------------------------------------------------------

class TestBacktestEngineRun:
    def test_single_strategy_returns_one_result(self, cost_data, reserved_strategy):
        engine = BacktestEngine(cost_data=cost_data)
        results = engine.run([reserved_strategy])
        assert len(results) == 1

    def test_multiple_strategies(self, cost_data, reserved_strategy, spot_strategy):
        engine = BacktestEngine(cost_data=cost_data)
        results = engine.run([reserved_strategy, spot_strategy])
        assert len(results) == 2

    def test_result_strategy_matches(self, cost_data, reserved_strategy):
        engine = BacktestEngine(cost_data=cost_data)
        result = engine.run([reserved_strategy])[0]
        assert result.strategy.name == "Reserved Instances"

    def test_optimised_costs_are_lower(self, cost_data, reserved_strategy):
        engine = BacktestEngine(cost_data=cost_data)
        result = engine.run([reserved_strategy])[0]
        assert result.optimized_total < result.original_total

    def test_spot_cheaper_than_reserved(self, cost_data, reserved_strategy, spot_strategy):
        engine = BacktestEngine(cost_data=cost_data)
        res_ri, res_spot = engine.run([reserved_strategy, spot_strategy])
        assert res_spot.optimized_total < res_ri.optimized_total


# ---------------------------------------------------------------------------
# Metrics validation
# ---------------------------------------------------------------------------

class TestBacktestMetrics:
    def test_savings_rate_positive(self, cost_data, reserved_strategy):
        engine = BacktestEngine(cost_data=cost_data)
        result = engine.run([reserved_strategy])[0]
        assert result.metrics.savings_rate == pytest.approx(30.0, abs=1e-3)

    def test_cost_efficiency_greater_than_one(self, cost_data, reserved_strategy):
        engine = BacktestEngine(cost_data=cost_data)
        result = engine.run([reserved_strategy])[0]
        assert result.metrics.cost_efficiency > 1.0

    def test_cumulative_savings_positive(self, cost_data, reserved_strategy):
        engine = BacktestEngine(cost_data=cost_data)
        result = engine.run([reserved_strategy])[0]
        assert result.metrics.cumulative_savings > 0.0

    def test_mean_cost_positive(self, cost_data, reserved_strategy):
        engine = BacktestEngine(cost_data=cost_data)
        result = engine.run([reserved_strategy])[0]
        assert result.metrics.mean_cost > 0.0

    def test_trend_slope_positive_for_increasing_data(self, cost_data, reserved_strategy):
        # Data has a positive cost trend; slope should be positive after optimisation
        engine = BacktestEngine(cost_data=cost_data)
        result = engine.run([reserved_strategy])[0]
        assert result.metrics.trend_slope > 0.0

    def test_trend_slope_zero_for_flat_data(self, flat_cost_data, reserved_strategy):
        engine = BacktestEngine(cost_data=flat_cost_data)
        result = engine.run([reserved_strategy])[0]
        assert result.metrics.trend_slope == pytest.approx(0.0, abs=1e-6)

    def test_anomalies_is_list(self, cost_data, reserved_strategy):
        engine = BacktestEngine(cost_data=cost_data)
        result = engine.run([reserved_strategy])[0]
        assert isinstance(result.metrics.anomalies, list)

    def test_flat_data_no_anomalies(self, flat_cost_data, reserved_strategy):
        engine = BacktestEngine(cost_data=flat_cost_data)
        result = engine.run([reserved_strategy])[0]
        assert len(result.metrics.anomalies) == 0

    def test_metrics_as_dict(self, cost_data, reserved_strategy):
        engine = BacktestEngine(cost_data=cost_data)
        result = engine.run([reserved_strategy])[0]
        d = result.metrics.as_dict()
        assert "savings_rate" in d
        assert "roi" in d
        assert "cumulative_savings" in d
        assert isinstance(d["anomalies"], list)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestBacktestEdgeCases:
    def test_empty_cost_data_raises(self, reserved_strategy):
        engine = BacktestEngine(cost_data=CostData(entries=[]))
        with pytest.raises(ValueError, match="no entries"):
            engine.run([reserved_strategy])

    def test_single_entry(self, reserved_strategy):
        data = CostData(
            entries=[CostEntry(date(2024, 1, 1), "Lambda", 200.0, 1.0, "us-west-2")]
        )
        engine = BacktestEngine(cost_data=data)
        result = engine.run([reserved_strategy])[0]
        assert result.metrics.savings_rate == pytest.approx(30.0, abs=1e-3)

    def test_no_savings_strategy(self, cost_data):
        no_op = Strategy(
            name="No-op",
            description="Does nothing",
            apply=lambda e: e.cost,
        )
        engine = BacktestEngine(cost_data=cost_data)
        result = engine.run([no_op])[0]
        assert result.metrics.savings_rate == pytest.approx(0.0, abs=1e-6)
        assert result.metrics.cumulative_savings == pytest.approx(0.0, abs=1e-6)

    def test_custom_anomaly_threshold(self, flat_cost_data):
        """A very low threshold should flag many anomaly periods."""
        noisy_strategy = Strategy(
            name="Noisy",
            description="adds variation",
            # inject one large cost spike
            apply=lambda e: e.cost * (3.0 if e.date.month == 6 else 1.0),
        )
        engine = BacktestEngine(
            cost_data=flat_cost_data,
            anomaly_threshold=1.5,
        )
        result = engine.run([noisy_strategy])[0]
        # The spike in June should be flagged
        assert len(result.metrics.anomalies) >= 1

    def test_result_summary_contains_strategy_name(self, cost_data, reserved_strategy):
        engine = BacktestEngine(cost_data=cost_data)
        result = engine.run([reserved_strategy])[0]
        summary = result.summary()
        assert "Reserved Instances" in summary
        assert "Savings" in summary


# ---------------------------------------------------------------------------
# CostData helpers
# ---------------------------------------------------------------------------

class TestCostData:
    def test_total_cost(self, cost_data):
        assert cost_data.total_cost() == pytest.approx(sum(e.cost for e in cost_data.entries))

    def test_costs_length(self, cost_data):
        assert len(cost_data.costs()) == len(cost_data.entries)

    def test_filter_by_service(self):
        entries = [
            CostEntry(date(2024, 1, 1), "EC2", 100.0, 10.0, "us-east-1"),
            CostEntry(date(2024, 1, 1), "S3", 50.0, 200.0, "us-east-1"),
            CostEntry(date(2024, 2, 1), "EC2", 120.0, 12.0, "us-east-1"),
        ]
        data = CostData(entries=entries)
        ec2_data = data.filter_by_service("EC2")
        assert len(ec2_data.entries) == 2
        assert all(e.service == "EC2" for e in ec2_data.entries)

    def test_filter_by_region(self):
        entries = [
            CostEntry(date(2024, 1, 1), "EC2", 100.0, 10.0, "us-east-1"),
            CostEntry(date(2024, 1, 1), "EC2", 80.0, 8.0, "eu-west-1"),
        ]
        data = CostData(entries=entries)
        us_data = data.filter_by_region("us-east-1")
        assert len(us_data.entries) == 1


# ---------------------------------------------------------------------------
# compare() requires LLM client
# ---------------------------------------------------------------------------

class TestCompare:
    def test_compare_without_llm_raises(self, cost_data, reserved_strategy):
        engine = BacktestEngine(cost_data=cost_data)
        with pytest.raises(RuntimeError, match="LLM client"):
            engine.compare([reserved_strategy])
