"""
FinOps Backtest Engine.

The :class:`BacktestEngine` is the central orchestrator.  It accepts historical
cost data and a list of optimisation strategies, runs each strategy against the
data, computes all financial metrics via the C/CFFI metrics library, and
optionally enriches results with LLM-generated insights.

Typical usage::

    from datetime import date
    from finops_backtest import BacktestEngine
    from finops_backtest.data import CostData, CostEntry, Strategy

    entries = [
        CostEntry(date(2024, 1, i+1), "EC2", 100.0 + i*5, 10.0, "us-east-1")
        for i in range(12)
    ]
    data = CostData(entries=entries)

    reserved = Strategy(
        name="Reserved Instances",
        description="30% discount via 1-year RI commitment.",
        apply=lambda e: e.cost * 0.70,
    )

    engine = BacktestEngine(cost_data=data)
    results = engine.run([reserved])
    print(results[0].summary())
"""

from __future__ import annotations

import logging
from typing import List, Optional

from finops_backtest.data.models import (
    BacktestResult,
    CostData,
    Strategy,
    StrategyMetrics,
)
from finops_backtest.metrics import (
    anomaly_score,
    array_mean,
    array_stddev,
    cost_efficiency,
    cumulative_savings,
    payback_period,
    roi,
    savings_rate,
    trend_slope,
)

logger = logging.getLogger(__name__)

# Periods with |z-score| above this threshold are flagged as anomalies.
DEFAULT_ANOMALY_THRESHOLD = 2.0

# Default assumption: optimisation requires upfront investment equal to 1×
# the monthly saving, used when no explicit investment is provided.
DEFAULT_INVESTMENT_MONTHS = 1


class BacktestEngine:
    """Runs optimisation strategies against historical cost data.

    Args:
        cost_data:          Historical cost records to backtest against.
        llm_client:         Optional :class:`~finops_backtest.llm.LLMClient`
                            used to generate natural-language insights for each
                            result.  When ``None`` (default), the ``insights``
                            field of each result is left empty.
        anomaly_threshold:  Z-score threshold above which a period is flagged
                            as anomalous (default: 2.0 = 2-sigma rule).
        investment_months:  Number of monthly savings used as a proxy for the
                            upfront investment when calculating ROI and payback
                            period (default: 1).
    """

    def __init__(
        self,
        cost_data: CostData,
        llm_client=None,
        anomaly_threshold: float = DEFAULT_ANOMALY_THRESHOLD,
        investment_months: int = DEFAULT_INVESTMENT_MONTHS,
    ) -> None:
        self.cost_data = cost_data
        self.llm_client = llm_client
        self.anomaly_threshold = anomaly_threshold
        self.investment_months = investment_months

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(self, strategies: List[Strategy]) -> List[BacktestResult]:
        """Run *strategies* against the engine's cost data.

        Args:
            strategies: One or more :class:`~finops_backtest.data.Strategy`
                        objects to evaluate.

        Returns:
            A list of :class:`~finops_backtest.data.BacktestResult` objects,
            one per strategy, in the same order as the input.
        """
        results: List[BacktestResult] = []
        for strategy in strategies:
            logger.debug("Running strategy: %s", strategy.name)
            result = self._run_strategy(strategy)
            if self.llm_client is not None:
                try:
                    result.insights = self.llm_client.analyze_result(result)
                except Exception as exc:  # pragma: no cover
                    logger.warning(
                        "LLM analysis failed for strategy '%s': %s",
                        strategy.name,
                        exc,
                    )
            results.append(result)
        return results

    def compare(self, strategies: List[Strategy]) -> str:
        """Run all strategies and return a comparative LLM analysis.

        Requires an LLM client to be configured.  Raises :exc:`RuntimeError`
        if no ``llm_client`` was supplied.

        Args:
            strategies: Strategies to compare.

        Returns:
            LLM-generated comparative analysis as a plain string.
        """
        if self.llm_client is None:
            raise RuntimeError(
                "An LLM client is required for compare(). "
                "Pass llm_client= when constructing BacktestEngine."
            )
        results = [self._run_strategy(s) for s in strategies]
        return self.llm_client.analyze_results(results)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_strategy(self, strategy: Strategy) -> BacktestResult:
        """Apply *strategy* to every cost entry and compute metrics."""
        entries = self.cost_data.entries
        if not entries:
            raise ValueError("CostData contains no entries.")

        original: List[float] = [e.cost for e in entries]
        optimized: List[float] = [strategy.apply(e) for e in entries]

        metrics = self._compute_metrics(original, optimized)

        return BacktestResult(
            strategy=strategy,
            original_costs=original,
            optimized_costs=optimized,
            metrics=metrics,
        )

    def _compute_metrics(
        self,
        original: List[float],
        optimized: List[float],
    ) -> StrategyMetrics:
        """Compute all :class:`StrategyMetrics` for one strategy run."""
        total_orig = sum(original)
        total_opt = sum(optimized)

        s_rate = savings_rate(total_orig, total_opt)
        c_eff = cost_efficiency(total_opt, total_orig)

        # Proxy investment = investment_months × mean monthly saving
        mean_saving = (total_orig - total_opt) / max(len(original), 1)
        investment = self.investment_months * max(mean_saving, 0.0)
        total_savings = total_orig - total_opt

        r = roi(total_savings, investment)
        pb = payback_period(investment, mean_saving)

        mean_c = array_mean(optimized)
        stddev_c = array_stddev(optimized)
        slope = trend_slope(optimized)
        cum_savings = cumulative_savings(original, optimized)

        # Detect anomalous periods
        anomalies = []
        for i, c in enumerate(optimized):
            z = anomaly_score(c, mean_c, stddev_c)
            if abs(z) > self.anomaly_threshold:
                anomalies.append((i, z))

        return StrategyMetrics(
            savings_rate=s_rate,
            cost_efficiency=c_eff,
            roi=r,
            payback_months=pb,
            mean_cost=mean_c,
            cost_stddev=stddev_c,
            trend_slope=slope,
            cumulative_savings=cum_savings,
            anomalies=anomalies,
        )
