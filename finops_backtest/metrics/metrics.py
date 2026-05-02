"""
Python wrapper around the C metrics CFFI extension.

Provides a :class:`Metrics` facade with a clean Python API, plus module-level
convenience functions so callers do not need to manage the FFI library object.

If the compiled C extension is not available, a pure-Python fallback is used
automatically so the package can still be imported and tested without a C
compiler.
"""

from __future__ import annotations

import math
from typing import Sequence


# ---------------------------------------------------------------------------
# Try to import the compiled C extension; fall back to pure Python otherwise.
# ---------------------------------------------------------------------------

def _load_cffi_lib():
    try:
        from finops_backtest.metrics._metrics_cffi import ffi, lib  # type: ignore
        return ffi, lib
    except ImportError:
        return None, None


_ffi, _lib = _load_cffi_lib()
_USE_C = _lib is not None


# ---------------------------------------------------------------------------
# Pure-Python fallback implementations (used when C extension is unavailable)
# ---------------------------------------------------------------------------

def _py_savings_rate(original_cost: float, optimized_cost: float) -> float:
    if original_cost <= 0.0:
        return 0.0
    return ((original_cost - optimized_cost) / original_cost) * 100.0


def _py_cost_efficiency(actual_cost: float, baseline_cost: float) -> float:
    if actual_cost <= 0.0:
        return 0.0
    return baseline_cost / actual_cost


def _py_roi(total_savings: float, total_investment: float) -> float:
    if total_investment <= 0.0:
        return 0.0
    return ((total_savings - total_investment) / total_investment) * 100.0


def _py_payback_period(investment: float, monthly_savings: float) -> float:
    if monthly_savings <= 0.0:
        return -1.0
    return investment / monthly_savings


def _py_array_mean(values: Sequence[float]) -> float:
    n = len(values)
    if n == 0:
        return 0.0
    return sum(values) / n


def _py_array_variance(values: Sequence[float]) -> float:
    n = len(values)
    if n == 0:
        return 0.0
    mean = _py_array_mean(values)
    return sum((v - mean) ** 2 for v in values) / n


def _py_array_stddev(values: Sequence[float]) -> float:
    return math.sqrt(_py_array_variance(values))


def _py_anomaly_score(cost: float, mean: float, stddev: float) -> float:
    if stddev <= 0.0:
        return 0.0
    return (cost - mean) / stddev


def _py_trend_slope(values: Sequence[float]) -> float:
    n = len(values)
    if n <= 1:
        return 0.0
    sum_x = sum_y = sum_xy = sum_x2 = 0.0
    for i, v in enumerate(values):
        sum_x += i
        sum_y += v
        sum_xy += i * v
        sum_x2 += i * i
    denom = n * sum_x2 - sum_x * sum_x
    if denom == 0.0:
        return 0.0
    return (n * sum_xy - sum_x * sum_y) / denom


def _py_cumulative_savings(original: Sequence[float], optimized: Sequence[float]) -> float:
    return sum(o - p for o, p in zip(original, optimized))


# ---------------------------------------------------------------------------
# Public functions (dispatch to C or Python depending on availability)
# ---------------------------------------------------------------------------

def _to_c_array(values: Sequence[float]):
    """Convert a Python sequence to a C double array for CFFI calls."""
    arr = _ffi.new("double[]", len(values))
    for i, v in enumerate(values):
        arr[i] = float(v)
    return arr, len(values)


def savings_rate(original_cost: float, optimized_cost: float) -> float:
    """Return savings achieved as a percentage of *original_cost*.

    >>> savings_rate(100.0, 75.0)
    25.0
    """
    if _USE_C:
        return _lib.savings_rate(original_cost, optimized_cost)
    return _py_savings_rate(original_cost, optimized_cost)


def cost_efficiency(actual_cost: float, baseline_cost: float) -> float:
    """Return the ratio of *baseline_cost* to *actual_cost*.

    A value greater than 1 means the actual cost is lower than baseline.

    >>> cost_efficiency(80.0, 100.0)
    1.25
    """
    if _USE_C:
        return _lib.cost_efficiency(actual_cost, baseline_cost)
    return _py_cost_efficiency(actual_cost, baseline_cost)


def roi(total_savings: float, total_investment: float) -> float:
    """Return Return-on-Investment as a percentage.

    >>> round(roi(150.0, 100.0), 1)
    50.0
    """
    if _USE_C:
        return _lib.roi(total_savings, total_investment)
    return _py_roi(total_savings, total_investment)


def payback_period(investment: float, monthly_savings: float) -> float:
    """Return the payback period in months.

    Returns ``-1`` when *monthly_savings* is zero or negative.

    >>> payback_period(1200.0, 100.0)
    12.0
    """
    if _USE_C:
        return _lib.payback_period(investment, monthly_savings)
    return _py_payback_period(investment, monthly_savings)


def array_mean(values: Sequence[float]) -> float:
    """Return the arithmetic mean of *values*.

    >>> array_mean([10.0, 20.0, 30.0])
    20.0
    """
    if _USE_C:
        arr, n = _to_c_array(values)
        return _lib.array_mean(arr, n)
    return _py_array_mean(values)


def array_variance(values: Sequence[float]) -> float:
    """Return the population variance of *values*.

    >>> array_variance([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])
    4.0
    """
    if _USE_C:
        arr, n = _to_c_array(values)
        return _lib.array_variance(arr, n)
    return _py_array_variance(values)


def array_stddev(values: Sequence[float]) -> float:
    """Return the population standard deviation of *values*.

    >>> array_stddev([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])
    2.0
    """
    if _USE_C:
        arr, n = _to_c_array(values)
        return _lib.array_stddev(arr, n)
    return _py_array_stddev(values)


def anomaly_score(cost: float, mean: float, stddev: float) -> float:
    """Return the z-score of *cost* relative to the given distribution.

    Values with ``|score| > 2`` are typically considered anomalies.

    >>> anomaly_score(150.0, 100.0, 25.0)
    2.0
    """
    if _USE_C:
        return _lib.anomaly_score(cost, mean, stddev)
    return _py_anomaly_score(cost, mean, stddev)


def trend_slope(values: Sequence[float]) -> float:
    """Return the OLS regression slope over a time series.

    A positive value indicates growing costs; negative means declining.

    >>> trend_slope([10.0, 20.0, 30.0])
    10.0
    """
    if _USE_C:
        arr, n = _to_c_array(values)
        return _lib.trend_slope(arr, n)
    return _py_trend_slope(values)


def cumulative_savings(original: Sequence[float], optimized: Sequence[float]) -> float:
    """Return total savings summed across all periods.

    >>> cumulative_savings([100.0, 120.0, 110.0], [80.0, 90.0, 85.0])
    75.0
    """
    if _USE_C:
        orig_arr, n = _to_c_array(original)
        opt_arr, _ = _to_c_array(optimized)
        return _lib.cumulative_savings(orig_arr, opt_arr, n)
    return _py_cumulative_savings(original, optimized)


# ---------------------------------------------------------------------------
# Convenience: expose whether the C extension is active
# ---------------------------------------------------------------------------

def using_c_extension() -> bool:
    """Return ``True`` if the compiled C extension is being used."""
    return _USE_C
