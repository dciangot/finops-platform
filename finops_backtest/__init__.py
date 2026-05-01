"""
finops_backtest
~~~~~~~~~~~~~~~

FinOps Backtest Engine — cloud cost optimisation analysis with C-backed metrics
and optional LLM insights.

Quick start::

    from datetime import date
    from finops_backtest import BacktestEngine
    from finops_backtest.data import CostData, CostEntry, Strategy

    entries = [
        CostEntry(date(2024, i + 1, 1), "EC2", 1000.0 + i * 50, 20.0, "us-east-1")
        for i in range(12)
    ]
    data = CostData(entries=entries)

    ri_strategy = Strategy(
        name="Reserved Instances",
        description="1-year RI commitment with ~30% discount.",
        apply=lambda e: e.cost * 0.70,
    )

    engine = BacktestEngine(cost_data=data)
    results = engine.run([ri_strategy])
    print(results[0].summary())
"""

from finops_backtest.engine import BacktestEngine
from finops_backtest.data.models import (
    CostEntry,
    CostData,
    Strategy,
    StrategyMetrics,
    BacktestResult,
)

__all__ = [
    "BacktestEngine",
    "CostEntry",
    "CostData",
    "Strategy",
    "StrategyMetrics",
    "BacktestResult",
]

__version__ = "0.1.0"
