# finops-platform

## FinOps Backtest Engine

A cloud cost optimisation backtest engine with:

- **C-backed metrics** via [CFFI](https://cffi.readthedocs.io/) for high-performance financial calculations
- **LLM insights** via any OpenAI-compatible endpoint (local or cloud), supporting [Ollama](https://ollama.com), [LM Studio](https://lmstudio.ai), and OpenAI
- **Pluggable strategies** – express any optimisation as a simple Python callable

---

## Quick Start

```python
from datetime import date
from finops_backtest import BacktestEngine, CostData, CostEntry, Strategy

# 1. Load historical cost data
entries = [
    CostEntry(date(2024, i + 1, 1), "EC2", 1000.0 + i * 50, 20.0, "us-east-1")
    for i in range(12)
]
data = CostData(entries=entries)

# 2. Define optimisation strategies
strategies = [
    Strategy(
        name="Reserved Instances (30% off)",
        description="Commit to 1-year Reserved Instances.",
        apply=lambda e: e.cost * 0.70,
    ),
    Strategy(
        name="Spot Instances (60% off)",
        description="Use Spot Instances for fault-tolerant workloads.",
        apply=lambda e: e.cost * 0.40,
    ),
]

# 3. Run the backtest
engine = BacktestEngine(cost_data=data)
results = engine.run(strategies)

for r in results:
    print(r.summary())
```

### With LLM Insights (local Ollama)

```python
from finops_backtest.llm import LLMClient

llm = LLMClient(
    base_url="http://localhost:11434/v1",  # Ollama local endpoint
    api_key="ollama",
    model="llama3",
)

engine = BacktestEngine(cost_data=data, llm_client=llm)
results = engine.run(strategies)

print(results[0].insights)   # LLM-generated analysis
```

### With OpenAI

```python
llm = LLMClient(
    base_url="https://api.openai.com/v1",
    api_key="sk-...",
    model="gpt-4o-mini",
)
```

---

## MCP Server

The engine ships with a built-in [Model Context Protocol](https://modelcontextprotocol.io/) (MCP) server that lets AI assistants run backtests directly.

### Start the server

```bash
# Via the dedicated console script
finops-backtest-mcp

# Or via the module CLI
python -m finops_backtest mcp
```

Both commands start an MCP server on the **stdio** transport (the default for most MCP clients).

### Claude Desktop configuration

Add the following block to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "finops-backtest": {
      "command": "finops-backtest-mcp"
    }
  }
}
```

### Available MCP tools

| Tool | Description |
|---|---|
| `list_builtin_strategies` | List all built-in strategy types with descriptions |
| `run_backtest` | Run one or more strategies against historical cost data and return per-strategy metrics |

### Tool input format

**`run_backtest`**

```json
{
  "cost_entries": [
    {
      "date": "2024-01-01",
      "service": "EC2",
      "cost": 1000.0,
      "usage": 20.0,
      "region": "us-east-1",
      "tags": {"env": "production"}
    }
  ],
  "strategies": [
    {"type": "reserved_instances"},
    {"type": "spot_instances"},
    {"type": "custom_discount", "discount_pct": 25.0, "name": "Negotiated Discount"}
  ]
}
```

Built-in strategy types: `reserved_instances` (30 % off), `spot_instances` (60 % off), `rightsizing` (15 % off), `custom_discount` (any `discount_pct`).

### Installation with MCP support

```bash
pip install ".[mcp]"
```

---

```bash
python -m finops_backtest demo
```

---

## Architecture

```
finops_backtest/
├── engine.py            # BacktestEngine orchestrator
├── __main__.py          # CLI entry point
├── mcp_server.py        # MCP server (FastMCP tools)
├── data/
│   └── models.py        # CostEntry, CostData, Strategy, BacktestResult
├── llm/
│   └── client.py        # OpenAI-compatible LLM client
└── metrics/
    ├── c_metrics.c      # C implementation of all metrics
    ├── c_metrics.h      # C header
    ├── ffi_build.py     # CFFI build script
    └── metrics.py       # Python wrapper (C + pure-Python fallback)
```

### Metrics (C via CFFI)

| Function | Description |
|---|---|
| `savings_rate(original, optimized)` | Percentage of costs saved |
| `cost_efficiency(actual, baseline)` | Baseline/actual ratio |
| `roi(savings, investment)` | Return on investment (%) |
| `payback_period(investment, monthly_savings)` | Break-even in months |
| `array_mean(values)` | Arithmetic mean |
| `array_variance(values)` | Population variance |
| `array_stddev(values)` | Population standard deviation |
| `anomaly_score(cost, mean, stddev)` | Z-score anomaly detection |
| `trend_slope(values)` | OLS regression slope over time |
| `cumulative_savings(original, optimized)` | Total monetary savings |

---

## Installation

```bash
pip install .
```

The C extension is built automatically during installation.  To build it
manually (useful during development):

```bash
python finops_backtest/metrics/ffi_build.py
```

### Requirements

- Python ≥ 3.10
- `cffi >= 1.15`
- `openai >= 1.0` (for LLM features)
- A C compiler (GCC / Clang) for building the metrics extension

---

## Testing

```bash
pip install ".[dev]"
pytest
```
