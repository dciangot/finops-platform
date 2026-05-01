"""Tests for the LLMClient."""

from __future__ import annotations

from unittest.mock import MagicMock, patch
from datetime import date

import pytest

from finops_backtest.llm import LLMClient
from finops_backtest.data.models import (
    BacktestResult,
    CostEntry,
    Strategy,
    StrategyMetrics,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_result(strategy_name: str = "Test Strategy") -> BacktestResult:
    strategy = Strategy(
        name=strategy_name,
        description="A test strategy",
        apply=lambda e: e.cost * 0.70,
    )
    metrics = StrategyMetrics(
        savings_rate=30.0,
        cost_efficiency=1.4286,
        roi=200.0,
        payback_months=1.0,
        mean_cost=700.0,
        cost_stddev=0.0,
        trend_slope=0.0,
        cumulative_savings=3600.0,
        anomalies=[],
    )
    return BacktestResult(
        strategy=strategy,
        original_costs=[1000.0] * 12,
        optimized_costs=[700.0] * 12,
        metrics=metrics,
    )


def _mock_openai_client(response_text: str):
    """Return a mock openai.OpenAI client whose completions return *response_text*."""
    mock_client = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = response_text
    mock_client.chat.completions.create.return_value = MagicMock(
        choices=[mock_choice]
    )
    return mock_client


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestLLMClientInit:
    def test_default_base_url(self):
        client = LLMClient()
        assert client.base_url == "https://api.openai.com/v1"

    def test_custom_base_url(self):
        client = LLMClient(base_url="http://localhost:11434/v1", api_key="ollama")
        assert client.base_url == "http://localhost:11434/v1"

    def test_model_attribute(self):
        client = LLMClient(model="llama3")
        assert client.model == "llama3"


# ---------------------------------------------------------------------------
# analyze_result
# ---------------------------------------------------------------------------

class TestAnalyzeResult:
    def test_returns_string(self):
        client = LLMClient()
        client._client = _mock_openai_client("Great savings!")
        result = _make_result()
        insights = client.analyze_result(result)
        assert isinstance(insights, str)
        assert insights == "Great savings!"

    def test_calls_chat_completions(self):
        client = LLMClient()
        mock_inner = _mock_openai_client("Some insight")
        client._client = mock_inner
        result = _make_result()
        client.analyze_result(result)
        mock_inner.chat.completions.create.assert_called_once()

    def test_prompt_contains_strategy_name(self):
        client = LLMClient()
        mock_inner = _mock_openai_client("ok")
        client._client = mock_inner
        result = _make_result("Reserved Instances")
        client.analyze_result(result)
        call_kwargs = mock_inner.chat.completions.create.call_args
        messages = call_kwargs[1]["messages"] if call_kwargs[1] else call_kwargs[0][1]
        user_msg = next(m["content"] for m in messages if m["role"] == "user")
        assert "Reserved Instances" in user_msg

    def test_prompt_contains_savings_rate(self):
        client = LLMClient()
        mock_inner = _mock_openai_client("ok")
        client._client = mock_inner
        result = _make_result()
        client.analyze_result(result)
        call_kwargs = mock_inner.chat.completions.create.call_args
        messages = call_kwargs[1]["messages"] if call_kwargs[1] else call_kwargs[0][1]
        user_msg = next(m["content"] for m in messages if m["role"] == "user")
        assert "30" in user_msg  # savings rate is 30%


# ---------------------------------------------------------------------------
# analyze_results (comparison)
# ---------------------------------------------------------------------------

class TestAnalyzeResults:
    def test_returns_string(self):
        client = LLMClient()
        client._client = _mock_openai_client("Strategy A is best")
        results = [_make_result("A"), _make_result("B")]
        comparison = client.analyze_results(results)
        assert isinstance(comparison, str)

    def test_prompt_contains_all_strategy_names(self):
        client = LLMClient()
        mock_inner = _mock_openai_client("comparison result")
        client._client = mock_inner
        results = [_make_result("Strategy Alpha"), _make_result("Strategy Beta")]
        client.analyze_results(results)
        call_kwargs = mock_inner.chat.completions.create.call_args
        messages = call_kwargs[1]["messages"] if call_kwargs[1] else call_kwargs[0][1]
        user_msg = next(m["content"] for m in messages if m["role"] == "user")
        assert "Strategy Alpha" in user_msg
        assert "Strategy Beta" in user_msg


# ---------------------------------------------------------------------------
# ask
# ---------------------------------------------------------------------------

class TestAsk:
    def test_basic_question(self):
        client = LLMClient()
        client._client = _mock_openai_client("The answer is 42")
        answer = client.ask("What is the meaning of life?")
        assert answer == "The answer is 42"

    def test_question_with_context(self):
        client = LLMClient()
        mock_inner = _mock_openai_client("Use Reserved Instances")
        client._client = mock_inner
        client.ask("What should I do?", context="Monthly cost is $10,000")
        call_kwargs = mock_inner.chat.completions.create.call_args
        messages = call_kwargs[1]["messages"] if call_kwargs[1] else call_kwargs[0][1]
        user_msg = next(m["content"] for m in messages if m["role"] == "user")
        assert "Monthly cost" in user_msg
        assert "What should I do?" in user_msg

    def test_exception_propagates(self):
        client = LLMClient()
        mock_inner = MagicMock()
        mock_inner.chat.completions.create.side_effect = RuntimeError("API error")
        client._client = mock_inner
        with pytest.raises(RuntimeError, match="API error"):
            client.ask("Will this fail?")


# ---------------------------------------------------------------------------
# LLM integration with BacktestEngine
# ---------------------------------------------------------------------------

class TestLLMIntegrationWithEngine:
    def test_insights_populated_when_llm_provided(self):
        from finops_backtest import BacktestEngine, CostData, CostEntry, Strategy

        entries = [
            CostEntry(date(2024, i + 1, 1), "EC2", 1000.0, 20.0, "us-east-1")
            for i in range(6)
        ]
        data = CostData(entries=entries)
        strategy = Strategy(
            name="RI",
            description="30% off",
            apply=lambda e: e.cost * 0.70,
        )

        llm = LLMClient()
        llm._client = _mock_openai_client("Good savings potential!")

        engine = BacktestEngine(cost_data=data, llm_client=llm)
        results = engine.run([strategy])
        assert results[0].insights == "Good savings potential!"

    def test_insights_none_without_llm(self):
        from finops_backtest import BacktestEngine, CostData, CostEntry, Strategy

        entries = [
            CostEntry(date(2024, 1, i + 1), "EC2", 500.0, 10.0, "us-west-2")
            for i in range(3)
        ]
        data = CostData(entries=entries)
        strategy = Strategy("no-op", "nothing", lambda e: e.cost)
        engine = BacktestEngine(cost_data=data)
        results = engine.run([strategy])
        assert results[0].insights is None
