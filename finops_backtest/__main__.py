"""
Command-line entry point for the FinOps Backtest Engine.

Usage::

    python -m finops_backtest --help
    python -m finops_backtest demo
"""

from __future__ import annotations

import argparse
import sys
from datetime import date


def _run_demo() -> None:
    """Run a built-in demo backtest with synthetic cost data."""
    from finops_backtest import BacktestEngine, CostData, CostEntry, Strategy

    print("FinOps Backtest Engine — Demo\n" + "=" * 40)

    # Synthetic 12-month cost data with a slight upward trend
    entries = [
        CostEntry(
            date=date(2024, i + 1, 1),
            service="EC2",
            cost=1000.0 + i * 50.0,
            usage=20.0,
            region="us-east-1",
            tags={"env": "production"},
        )
        for i in range(12)
    ]
    data = CostData(entries=entries)

    strategies = [
        Strategy(
            name="Reserved Instances (30% off)",
            description="Commit to 1-year Reserved Instances for a ~30% discount.",
            apply=lambda e: e.cost * 0.70,
        ),
        Strategy(
            name="Spot Instances (60% off)",
            description="Use Spot Instances for fault-tolerant workloads (~60% discount).",
            apply=lambda e: e.cost * 0.40,
        ),
        Strategy(
            name="Rightsizing (15% off)",
            description="Downsize over-provisioned instances (~15% cost reduction).",
            apply=lambda e: e.cost * 0.85,
        ),
    ]

    from finops_backtest.metrics import using_c_extension

    print(f"C extension active: {using_c_extension()}\n")

    engine = BacktestEngine(cost_data=data)
    results = engine.run(strategies)

    for result in results:
        print(result.summary())
        print("-" * 40)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="finops-backtest",
        description="FinOps Backtest Engine CLI",
    )
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("demo", help="Run a built-in demo backtest")

    args = parser.parse_args(argv)

    if args.command == "demo":
        _run_demo()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
