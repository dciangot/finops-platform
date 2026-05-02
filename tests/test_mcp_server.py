"""Tests for the FinOps Backtest MCP server."""

from __future__ import annotations

import pytest

from finops_backtest.mcp_server import (
    _BUILTIN_STRATEGIES,
    _build_strategy,
    _parse_cost_entry,
    list_builtin_strategies,
    run_backtest,
)


# ---------------------------------------------------------------------------
# Helper fixtures
# ---------------------------------------------------------------------------

def _sample_entries(n: int = 3):
    return [
        {
            "date": f"2024-{i + 1:02d}-01",
            "service": "EC2",
            "cost": 1000.0 + i * 100,
            "usage": 20.0,
            "region": "us-east-1",
        }
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# _parse_cost_entry
# ---------------------------------------------------------------------------

class TestParseCostEntry:
    def test_full_entry(self):
        from datetime import date

        entry = _parse_cost_entry(
            {
                "date": "2024-03-15",
                "service": "S3",
                "cost": 250.0,
                "usage": 500.0,
                "region": "eu-west-1",
                "tags": {"env": "prod"},
            }
        )
        assert entry.date == date(2024, 3, 15)
        assert entry.service == "S3"
        assert entry.cost == pytest.approx(250.0)
        assert entry.usage == pytest.approx(500.0)
        assert entry.region == "eu-west-1"
        assert entry.tags == {"env": "prod"}

    def test_optional_fields_default(self):
        entry = _parse_cost_entry(
            {"date": "2024-01-01", "service": "Lambda", "cost": 10.0}
        )
        assert entry.usage == pytest.approx(0.0)
        assert entry.region == ""
        assert entry.tags == {}


# ---------------------------------------------------------------------------
# _build_strategy
# ---------------------------------------------------------------------------

class TestBuildStrategy:
    def test_reserved_instances(self):
        from finops_backtest import CostEntry
        from datetime import date

        s = _build_strategy({"type": "reserved_instances"})
        assert "Reserved" in s.name
        entry = CostEntry(date(2024, 1, 1), "EC2", 1000.0, 10.0, "us-east-1")
        assert s.apply(entry) == pytest.approx(700.0)

    def test_spot_instances(self):
        from finops_backtest import CostEntry
        from datetime import date

        s = _build_strategy({"type": "spot_instances"})
        entry = CostEntry(date(2024, 1, 1), "EC2", 1000.0, 10.0, "us-east-1")
        assert s.apply(entry) == pytest.approx(400.0)

    def test_rightsizing(self):
        from finops_backtest import CostEntry
        from datetime import date

        s = _build_strategy({"type": "rightsizing"})
        entry = CostEntry(date(2024, 1, 1), "EC2", 1000.0, 10.0, "us-east-1")
        assert s.apply(entry) == pytest.approx(850.0)

    def test_custom_discount(self):
        from finops_backtest import CostEntry
        from datetime import date

        s = _build_strategy({"type": "custom_discount", "discount_pct": 20.0})
        entry = CostEntry(date(2024, 1, 1), "EC2", 1000.0, 10.0, "us-east-1")
        assert s.apply(entry) == pytest.approx(800.0)

    def test_custom_name_override(self):
        s = _build_strategy(
            {"type": "reserved_instances", "name": "My RI Strategy"}
        )
        assert s.name == "My RI Strategy"

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError, match="Unknown strategy type"):
            _build_strategy({"type": "turbo_savings"})


# ---------------------------------------------------------------------------
# list_builtin_strategies tool
# ---------------------------------------------------------------------------

class TestListBuiltinStrategies:
    def test_returns_list(self):
        result = list_builtin_strategies()
        assert isinstance(result, list)

    def test_includes_all_builtin_types(self):
        result = list_builtin_strategies()
        types = {s["type"] for s in result}
        for key in _BUILTIN_STRATEGIES:
            assert key in types

    def test_includes_custom_discount(self):
        result = list_builtin_strategies()
        types = {s["type"] for s in result}
        assert "custom_discount" in types

    def test_each_entry_has_required_keys(self):
        for entry in list_builtin_strategies():
            assert "type" in entry
            assert "name" in entry
            assert "description" in entry
            assert "discount_pct" in entry


# ---------------------------------------------------------------------------
# run_backtest tool
# ---------------------------------------------------------------------------

class TestRunBacktest:
    def test_single_strategy_returns_one_result(self):
        results = run_backtest(
            cost_entries=_sample_entries(6),
            strategies=[{"type": "reserved_instances"}],
        )
        assert len(results) == 1

    def test_multiple_strategies(self):
        results = run_backtest(
            cost_entries=_sample_entries(6),
            strategies=[
                {"type": "reserved_instances"},
                {"type": "spot_instances"},
            ],
        )
        assert len(results) == 2

    def test_result_has_required_keys(self):
        result = run_backtest(
            cost_entries=_sample_entries(6),
            strategies=[{"type": "reserved_instances"}],
        )[0]
        for key in (
            "strategy_name",
            "strategy_description",
            "original_total",
            "optimized_total",
            "metrics",
            "summary",
        ):
            assert key in result

    def test_metrics_has_expected_fields(self):
        result = run_backtest(
            cost_entries=_sample_entries(6),
            strategies=[{"type": "reserved_instances"}],
        )[0]
        metrics = result["metrics"]
        for field in (
            "savings_rate",
            "roi",
            "cumulative_savings",
            "mean_cost",
            "anomalies",
        ):
            assert field in metrics

    def test_optimized_total_lower_than_original(self):
        result = run_backtest(
            cost_entries=_sample_entries(6),
            strategies=[{"type": "reserved_instances"}],
        )[0]
        assert result["optimized_total"] < result["original_total"]

    def test_savings_rate_matches_strategy(self):
        result = run_backtest(
            cost_entries=_sample_entries(6),
            strategies=[{"type": "reserved_instances"}],
        )[0]
        assert result["metrics"]["savings_rate"] == pytest.approx(30.0, abs=1e-3)

    def test_custom_discount_strategy(self):
        result = run_backtest(
            cost_entries=_sample_entries(6),
            strategies=[{"type": "custom_discount", "discount_pct": 50.0}],
        )[0]
        assert result["metrics"]["savings_rate"] == pytest.approx(50.0, abs=1e-3)

    def test_summary_contains_strategy_name(self):
        result = run_backtest(
            cost_entries=_sample_entries(6),
            strategies=[{"type": "spot_instances"}],
        )[0]
        assert "Spot" in result["summary"]

    def test_empty_cost_entries_raises(self):
        with pytest.raises(ValueError, match="cost_entries must not be empty"):
            run_backtest(
                cost_entries=[],
                strategies=[{"type": "reserved_instances"}],
            )

    def test_empty_strategies_raises(self):
        with pytest.raises(ValueError, match="strategies must not be empty"):
            run_backtest(
                cost_entries=_sample_entries(3),
                strategies=[],
            )

    def test_tags_preserved_in_processing(self):
        entries = [
            {
                "date": "2024-01-01",
                "service": "EC2",
                "cost": 500.0,
                "usage": 10.0,
                "region": "us-east-1",
                "tags": {"env": "prod", "team": "infra"},
            }
        ]
        result = run_backtest(
            cost_entries=entries,
            strategies=[{"type": "rightsizing"}],
        )[0]
        assert result["optimized_total"] == pytest.approx(500.0 * 0.85, abs=1e-6)
