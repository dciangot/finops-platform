"""
LLM client for the FinOps Backtest Engine.

Provides an :class:`LLMClient` that wraps any OpenAI-compatible API endpoint
(including local servers such as Ollama, LM Studio, or llama.cpp with an
OpenAI-compatible HTTP layer).  The client can generate natural-language
insights from backtest results and answer FinOps questions interactively.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from finops_backtest.data.models import BacktestResult

# Default system prompt for the FinOps assistant
_SYSTEM_PROMPT = (
    "You are a FinOps expert assistant specialised in cloud cost optimisation. "
    "When presented with backtest results you provide clear, actionable "
    "recommendations backed by the data.  Be concise and focus on the most "
    "impactful insights."
)


class LLMClient:
    """OpenAI-compatible LLM client for FinOps insight generation.

    The *base_url* parameter accepts any OpenAI-compatible endpoint, making it
    easy to swap between:

    * **OpenAI** (default):  ``https://api.openai.com/v1``
    * **Ollama**:            ``http://localhost:11434/v1``
    * **LM Studio**:         ``http://localhost:1234/v1``
    * **llama.cpp server**:  ``http://localhost:8080/v1``

    Example::

        # Use a local Ollama instance
        client = LLMClient(
            base_url="http://localhost:11434/v1",
            api_key="ollama",
            model="llama3",
        )

        insights = client.analyze_result(backtest_result)

    Args:
        base_url:    Base URL of the OpenAI-compatible API server.
        api_key:     API key (use any non-empty string for local servers).
        model:       Model identifier to use for completions.
        temperature: Sampling temperature (0 = deterministic).
        max_tokens:  Maximum tokens in the completion.
        timeout:     HTTP request timeout in seconds.
    """

    def __init__(
        self,
        base_url: str = "https://api.openai.com/v1",
        api_key: str = "sk-placeholder",
        model: str = "gpt-4o-mini",
        temperature: float = 0.2,
        max_tokens: int = 1024,
        timeout: float = 60.0,
    ) -> None:
        try:
            import openai  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "The 'openai' package is required for LLMClient. "
                "Install it with: pip install openai"
            ) from exc

        self._client = openai.OpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
        )
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.base_url = base_url

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def analyze_result(self, result: "BacktestResult") -> str:
        """Generate LLM insights for a single :class:`BacktestResult`.

        The method serialises the result metrics into a structured prompt and
        returns the model's analysis as a plain string.

        Args:
            result: The backtest result to analyse.

        Returns:
            Natural-language insights and recommendations.
        """
        prompt = self._build_analysis_prompt(result)
        return self._complete(prompt)

    def analyze_results(self, results: List["BacktestResult"]) -> str:
        """Generate a comparative LLM analysis across multiple results.

        Useful for comparing several strategies head-to-head and getting a
        ranking recommendation.

        Args:
            results: List of backtest results (one per strategy).

        Returns:
            Comparative natural-language analysis.
        """
        prompt = self._build_comparison_prompt(results)
        return self._complete(prompt)

    def ask(self, question: str, context: Optional[str] = None) -> str:
        """Ask an arbitrary FinOps question with optional context.

        Args:
            question: Free-form question to ask the model.
            context:  Optional context string prepended to the question.

        Returns:
            Model response as a plain string.
        """
        user_content = question
        if context:
            user_content = f"Context:\n{context}\n\nQuestion:\n{question}"
        return self._complete(user_content)

    # ------------------------------------------------------------------
    # Prompt builders
    # ------------------------------------------------------------------

    def _build_analysis_prompt(self, result: "BacktestResult") -> str:
        m = result.metrics
        lines = [
            f"Strategy: {result.strategy.name}",
            f"Description: {result.strategy.description}",
            "",
            "=== Backtest Metrics ===",
            f"Original total cost : {result.original_total:,.2f}",
            f"Optimised total cost: {result.optimized_total:,.2f}",
            f"Savings rate        : {m.savings_rate:.2f}%",
            f"Cost efficiency     : {m.cost_efficiency:.4f}",
            f"ROI                 : {m.roi:.2f}%",
            f"Payback period      : {m.payback_months:.1f} months"
            if m.payback_months >= 0
            else "Payback period      : N/A (no positive monthly savings)",
            f"Mean optimised cost : {m.mean_cost:.2f}",
            f"Cost std deviation  : {m.cost_stddev:.2f}",
            f"Cost trend slope    : {m.trend_slope:+.4f} per period",
            f"Cumulative savings  : {m.cumulative_savings:,.2f}",
            f"Anomalous periods   : {len(m.anomalies)}",
        ]
        if m.anomalies:
            lines.append(
                "Anomaly details: "
                + ", ".join(f"period {i} (z={z:.2f})" for i, z in m.anomalies[:5])
            )
        lines += [
            "",
            "Please provide:",
            "1. A brief assessment of this strategy's effectiveness.",
            "2. Key risks or drawbacks to consider.",
            "3. Concrete next steps for the engineering/finance team.",
        ]
        return "\n".join(lines)

    def _build_comparison_prompt(self, results: List["BacktestResult"]) -> str:
        lines = ["Comparative FinOps Backtest Analysis", "=" * 40, ""]
        for r in results:
            m = r.metrics
            lines += [
                f"Strategy: {r.strategy.name}",
                f"  Savings rate      : {m.savings_rate:.2f}%",
                f"  ROI               : {m.roi:.2f}%",
                f"  Payback (months)  : {m.payback_months:.1f}"
                if m.payback_months >= 0
                else "  Payback (months)  : N/A",
                f"  Cumulative savings: {m.cumulative_savings:,.2f}",
                f"  Trend slope       : {m.trend_slope:+.4f}",
                "",
            ]
        lines += [
            "Please:",
            "1. Rank these strategies from best to worst with justification.",
            "2. Identify which strategy has the best risk/reward profile.",
            "3. Recommend a combined approach if applicable.",
        ]
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _complete(self, user_message: str) -> str:
        """Send a chat completion request and return the response text."""
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            content = response.choices[0].message.content
            return content.strip() if content else ""
        except Exception as exc:
            logger.error("LLM request failed: %s", exc)
            raise
