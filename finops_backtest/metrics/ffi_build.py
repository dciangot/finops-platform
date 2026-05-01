"""
CFFI build script for the FinOps metrics C extension.

Run this script directly to compile the shared library::

    python ffi_build.py

Or let setuptools handle it automatically via ``cffi_modules`` in
``pyproject.toml``.  The compiled extension is named
``finops_backtest.metrics._metrics_cffi``.
"""

import os
from cffi import FFI

ffi = FFI()

# Public API declarations (must match c_metrics.h)
ffi.cdef(
    """
    double savings_rate(double original_cost, double optimized_cost);
    double cost_efficiency(double actual_cost, double baseline_cost);
    double roi(double total_savings, double total_investment);
    double payback_period(double investment, double monthly_savings);
    double array_mean(const double *values, int n);
    double array_variance(const double *values, int n);
    double array_stddev(const double *values, int n);
    double anomaly_score(double cost, double mean, double stddev);
    double trend_slope(const double *values, int n);
    double cumulative_savings(const double *original, const double *optimized, int n);
    """
)

_src_dir = os.path.dirname(os.path.abspath(__file__))

ffi.set_source(
    "finops_backtest.metrics._metrics_cffi",
    '#include "c_metrics.h"',
    sources=[os.path.join(_src_dir, "c_metrics.c")],
    include_dirs=[_src_dir],
    libraries=["m"],
)

if __name__ == "__main__":
    # Output to the project root so the compiled extension lands at the right
    # Python package path: finops_backtest/metrics/_metrics_cffi.<ext>
    _project_root = os.path.dirname(os.path.dirname(_src_dir))
    ffi.compile(tmpdir=_project_root, verbose=True)
