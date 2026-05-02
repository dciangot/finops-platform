"""
FinOps Backtest MCP Server.

Exposes the :class:`~finops_backtest.BacktestEngine` as an MCP (Model Context
Protocol) server so that AI assistants and other MCP clients can run cost
optimisation backtests, inspect metrics, and compare strategies.

Usage::

    # Start the server (stdio transport – default for MCP clients)
    python -m finops_backtest mcp

    # Or import and run programmatically
    from finops_backtest.mcp_server import mcp
    mcp.run()

Built-in strategy types
-----------------------
Pass one of these ``type`` values in each strategy spec dict:

* ``"reserved_instances"`` – 30 % discount (1-year RI commitment)
* ``"spot_instances"``     – 60 % discount (fault-tolerant workloads)
* ``"rightsizing"``        – 15 % discount (right-size over-provisioned VMs)
* ``"custom_discount"``    – arbitrary percentage via ``"discount_pct"`` key

Cost entry dict keys
--------------------
``date`` (ISO-8601 string), ``service``, ``cost`` (float), ``usage`` (float),
``region``, and an optional ``tags`` dict.
"""

from __future__ import annotations

from datetime import date as _date
from typing import Any, Dict, List

from mcp.server.fastmcp import FastMCP

from finops_backtest import BacktestEngine, CostData, CostEntry, Strategy

mcp = FastMCP(
    "FinOps Backtest",
    instructions=(
        "Use this server to run cloud cost optimisation backtests, compare "
        "strategies, and compute FinOps metrics against historical billing data."
    ),
)

# ---------------------------------------------------------------------------
# Built-in strategy registry
# ---------------------------------------------------------------------------

_BUILTIN_STRATEGIES: Dict[str, Dict[str, Any]] = {
    "reserved_instances": {
        "name": "Reserved Instances (30 % off)",
        "description": (
            "Commit to 1-year Reserved Instances for a ~30 % discount on "
            "compute costs."
        ),
        "discount_pct": 30.0,
    },
    "spot_instances": {
        "name": "Spot Instances (60 % off)",
        "description": (
            "Use Spot/Preemptible instances for fault-tolerant workloads – "
            "typically ~60 % cheaper than on-demand pricing."
        ),
        "discount_pct": 60.0,
    },
    "rightsizing": {
        "name": "Rightsizing (15 % off)",
        "description": (
            "Downsize over-provisioned instances to their optimal size – "
            "typically yields ~15 % cost reduction."
        ),
        "discount_pct": 15.0,
    },
}


def _build_strategy(spec: Dict[str, Any]) -> Strategy:
    """Convert a strategy spec dict into a :class:`Strategy` instance."""
    strategy_type = spec.get("type", "")
    if strategy_type in _BUILTIN_STRATEGIES:
        info = _BUILTIN_STRATEGIES[strategy_type]
        discount = info["discount_pct"] / 100.0
        return Strategy(
            name=spec.get("name", info["name"]),
            description=spec.get("description", info["description"]),
            apply=lambda e, d=discount: e.cost * (1.0 - d),
        )
    if strategy_type == "custom_discount":
        discount_pct = float(spec.get("discount_pct", 0.0))
        discount = discount_pct / 100.0
        name = spec.get("name", f"Custom Discount ({discount_pct:.1f} % off)")
        description = spec.get(
            "description",
            f"Apply a custom {discount_pct:.1f} % discount to all cost entries.",
        )
        return Strategy(
            name=name,
            description=description,
            apply=lambda e, d=discount: e.cost * (1.0 - d),
        )
    raise ValueError(
        f"Unknown strategy type {strategy_type!r}. "
        f"Valid types: {', '.join(list(_BUILTIN_STRATEGIES) + ['custom_discount'])}"
    )


def _parse_cost_entry(entry: Dict[str, Any]) -> CostEntry:
    """Parse a plain dict into a :class:`CostEntry`."""
    raw_date = entry["date"]
    if isinstance(raw_date, str):
        parsed_date = _date.fromisoformat(raw_date)
    else:
        parsed_date = raw_date
    return CostEntry(
        date=parsed_date,
        service=str(entry["service"]),
        cost=float(entry["cost"]),
        usage=float(entry.get("usage", 0.0)),
        region=str(entry.get("region", "")),
        tags=dict(entry.get("tags", {})),
    )


# ---------------------------------------------------------------------------
# MCP tools
# ---------------------------------------------------------------------------


@mcp.tool()
def list_builtin_strategies() -> List[Dict[str, Any]]:
    """List all built-in optimisation strategies available to the backtest engine.

    Returns a list of strategy descriptors, each containing:

    * ``type``         – identifier to use in ``run_backtest``
    * ``name``         – human-readable name
    * ``description``  – what the strategy does
    * ``discount_pct`` – default percentage discount applied to costs
    """
    result = []
    for strategy_type, info in _BUILTIN_STRATEGIES.items():
        result.append(
            {
                "type": strategy_type,
                "name": info["name"],
                "description": info["description"],
                "discount_pct": info["discount_pct"],
            }
        )
    # Also document the custom_discount type
    result.append(
        {
            "type": "custom_discount",
            "name": "Custom Discount",
            "description": (
                "Apply any arbitrary percentage discount. "
                'Requires a "discount_pct" key in the strategy spec.'
            ),
            "discount_pct": None,
        }
    )
    return result


@mcp.tool()
def run_backtest(
    cost_entries: List[Dict[str, Any]],
    strategies: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Run a FinOps backtest and return per-strategy results with metrics.

    Args:
        cost_entries: Historical cost records.  Each entry is a dict with:

            * ``date``    – ISO-8601 date string (e.g. ``"2024-01-01"``)
            * ``service`` – cloud service name (e.g. ``"EC2"``)
            * ``cost``    – monetary cost as a float
            * ``usage``   – usage quantity as a float
            * ``region``  – cloud region (e.g. ``"us-east-1"``)
            * ``tags``    – (optional) dict of string key/value tags

        strategies: One or more strategy specs to evaluate.  Each spec is a
            dict with at least a ``type`` key.  Valid types are
            ``"reserved_instances"``, ``"spot_instances"``,
            ``"rightsizing"``, or ``"custom_discount"`` (the latter also
            requires a ``"discount_pct"`` key).  You may override the
            auto-generated ``"name"`` and ``"description"`` fields in any
            spec.

    Returns:
        A list of result dicts, one per strategy, each containing:

        * ``strategy_name``        – name of the strategy
        * ``strategy_description`` – description of the strategy
        * ``original_total``       – total original cost
        * ``optimized_total``      – total projected cost after optimisation
        * ``metrics``              – dict of computed FinOps metrics
        * ``summary``              – human-readable result summary

    Example input::

        cost_entries = [
            {"date": "2024-01-01", "service": "EC2", "cost": 1000.0,
             "usage": 20.0, "region": "us-east-1"},
        ]
        strategies = [
            {"type": "reserved_instances"},
            {"type": "custom_discount", "discount_pct": 25.0},
        ]
    """
    if not cost_entries:
        raise ValueError("cost_entries must not be empty.")
    if not strategies:
        raise ValueError("strategies must not be empty.")

    entries = [_parse_cost_entry(e) for e in cost_entries]
    strategy_objects = [_build_strategy(s) for s in strategies]

    engine = BacktestEngine(cost_data=CostData(entries=entries))
    results = engine.run(strategy_objects)

    output = []
    for result in results:
        output.append(
            {
                "strategy_name": result.strategy.name,
                "strategy_description": result.strategy.description,
                "original_total": result.original_total,
                "optimized_total": result.optimized_total,
                "metrics": result.metrics.as_dict(),
                "summary": result.summary(),
            }
        )
    return output


def main() -> None:
    """Entry point for the ``finops-backtest-mcp`` console script.

    Starts the MCP server using the stdio transport (suitable for most MCP
    clients, including Claude Desktop and the MCP Inspector).
    """
    mcp.run(transport="stdio")
