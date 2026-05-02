"""
Data models for the FinOps Backtest Engine.

These lightweight dataclasses represent the core domain objects used throughout
the engine: cost records, collections of cost data, optimisation strategies,
and backtest results.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any, Callable, Dict, List, Optional


@dataclass
class CostEntry:
    """A single line-item cost record for a cloud service.

    Attributes:
        date:    The billing date of the entry.
        service: The cloud service name (e.g. ``"EC2"``, ``"S3"``).
        cost:    Monetary cost in *currency* units.
        usage:   Usage quantity (units depend on the service, e.g. vCPU-hours).
        region:  Cloud region identifier (e.g. ``"us-east-1"``).
        tags:    Arbitrary key/value tags for grouping and filtering.
    """

    date: date
    service: str
    cost: float
    usage: float
    region: str
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class CostData:
    """A collection of :class:`CostEntry` records.

    Attributes:
        entries:  Ordered list of cost entries (typically sorted by date).
        currency: ISO 4217 currency code, defaults to ``"USD"``.
    """

    entries: List[CostEntry]
    currency: str = "USD"

    def costs(self) -> List[float]:
        """Return a plain list of cost values in entry order."""
        return [e.cost for e in self.entries]

    def total_cost(self) -> float:
        """Return the sum of all cost entries."""
        return sum(e.cost for e in self.entries)

    def filter_by_service(self, service: str) -> "CostData":
        """Return a new :class:`CostData` containing only entries for *service*."""
        filtered = [e for e in self.entries if e.service == service]
        return CostData(entries=filtered, currency=self.currency)

    def filter_by_region(self, region: str) -> "CostData":
        """Return a new :class:`CostData` containing only entries for *region*."""
        filtered = [e for e in self.entries if e.region == region]
        return CostData(entries=filtered, currency=self.currency)


@dataclass
class Strategy:
    """An optimisation strategy that transforms cost entries.

    The *apply* callable receives a single :class:`CostEntry` and returns the
    projected cost after applying the strategy.  This makes strategies fully
    composable and testable.

    Attributes:
        name:        Short human-readable strategy name.
        description: Detailed description of what the strategy does.
        apply:       A function ``(entry: CostEntry) -> float`` that returns
                     the projected cost for that entry.

    Example::

        reserved = Strategy(
            name="Reserved Instances",
            description="Apply a 30% discount via 1-year Reserved Instance commitment.",
            apply=lambda entry: entry.cost * 0.70,
        )
    """

    name: str
    description: str
    apply: Callable[[CostEntry], float]

    def __repr__(self) -> str:  # pragma: no cover
        return f"Strategy(name={self.name!r})"


@dataclass
class StrategyMetrics:
    """Quantitative metrics produced by running a strategy against historical data.

    Attributes:
        savings_rate:        Percentage of costs saved (0–100+).
        cost_efficiency:     Ratio of baseline to optimised cost (>1 is better).
        roi:                 Return on investment percentage.
        payback_months:      Months until investment breaks even, or ``-1``.
        mean_cost:           Mean period cost after optimisation.
        cost_stddev:         Population standard deviation of optimised costs.
        trend_slope:         OLS regression slope (positive = growing costs).
        cumulative_savings:  Total monetary savings over the backtest period.
        anomalies:           List of (index, z_score) tuples for periods where
                             ``|z_score| > anomaly_threshold``.
    """

    savings_rate: float
    cost_efficiency: float
    roi: float
    payback_months: float
    mean_cost: float
    cost_stddev: float
    trend_slope: float
    cumulative_savings: float
    anomalies: List[tuple[int, float]] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        """Return a plain ``dict`` representation (useful for JSON serialisation)."""
        return {
            "savings_rate": self.savings_rate,
            "cost_efficiency": self.cost_efficiency,
            "roi": self.roi,
            "payback_months": self.payback_months,
            "mean_cost": self.mean_cost,
            "cost_stddev": self.cost_stddev,
            "trend_slope": self.trend_slope,
            "cumulative_savings": self.cumulative_savings,
            "anomalies": [{"index": i, "z_score": z} for i, z in self.anomalies],
        }


@dataclass
class BacktestResult:
    """The result of running a single strategy against a :class:`CostData` set.

    Attributes:
        strategy:        The strategy that was evaluated.
        original_costs:  List of original cost values per period.
        optimized_costs: List of projected cost values after applying the strategy.
        metrics:         Computed :class:`StrategyMetrics`.
        insights:        Optional LLM-generated natural-language insights.
    """

    strategy: Strategy
    original_costs: List[float]
    optimized_costs: List[float]
    metrics: StrategyMetrics
    insights: Optional[str] = None

    @property
    def original_total(self) -> float:
        """Total original cost across all periods."""
        return sum(self.original_costs)

    @property
    def optimized_total(self) -> float:
        """Total optimised cost across all periods."""
        return sum(self.optimized_costs)

    def summary(self) -> str:
        """Return a concise human-readable summary of this result."""
        m = self.metrics
        lines = [
            f"Strategy : {self.strategy.name}",
            f"Original : {self.original_total:,.2f}",
            f"Optimised: {self.optimized_total:,.2f}",
            f"Savings  : {m.cumulative_savings:,.2f} ({m.savings_rate:.1f}%)",
            f"ROI      : {m.roi:.1f}%",
            f"Trend    : {m.trend_slope:+.2f} / period",
            f"Anomalies: {len(m.anomalies)}",
        ]
        if self.insights:
            lines.append(f"\nInsights:\n{self.insights}")
        return "\n".join(lines)
