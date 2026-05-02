"""
finops_backtest.metrics
~~~~~~~~~~~~~~~~~~~~~~~

FinOps metrics calculations backed by a C extension (via CFFI) with an
automatic pure-Python fallback.

Usage::

    from finops_backtest.metrics import savings_rate, array_mean, trend_slope

    rate = savings_rate(original_cost=1000.0, optimized_cost=750.0)
    # -> 25.0  (percent)
"""

from finops_backtest.metrics.metrics import (
    savings_rate,
    cost_efficiency,
    roi,
    payback_period,
    array_mean,
    array_variance,
    array_stddev,
    anomaly_score,
    trend_slope,
    cumulative_savings,
    using_c_extension,
)

__all__ = [
    "savings_rate",
    "cost_efficiency",
    "roi",
    "payback_period",
    "array_mean",
    "array_variance",
    "array_stddev",
    "anomaly_score",
    "trend_slope",
    "cumulative_savings",
    "using_c_extension",
]
