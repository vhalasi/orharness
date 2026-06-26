# ORHarness

> Bridge between large language models and operations research solvers.

**ORHarness** connects any LLM to [Google OR-Tools](https://developers.google.com/optimization)
through a multi-agent pipeline. Describe your problem in plain English — ORHarness parses it,
builds a model, generates solver code, runs it safely, and returns a plain-English answer.

## Status

**v0.1** — Core pipeline and public API are implemented. Scheduling is the most tested
problem type; routing and allocation are supported but less validated.

## Installation

From PyPI (once published):

```bash
pip install orharness
```

For local development:

```bash
git clone https://github.com/vhalasi/orharness
cd orharness
pip install -e ".[dev]"
cp .env.example .env   # add your ANTHROPIC_API_KEY (or other LiteLLM provider key)
```

## Quickstart

```python
from orharness import ORHarness

harness = ORHarness(model="claude-sonnet-4-6")
result = harness.solve(
    "Six nurses, three shifts per day, max 40 hours per week, "
    "at least two nurses per shift. Find a valid schedule."
)

print(result.success)             # harness completed end-to-end
print(result.feasible)            # solver found a usable solution
print(result.solution)            # plain English explanation
print(result.code)                # generated OR-Tools Python script
print(result.objective_value)     # None for feasibility-only problems
print(result.retries)             # debugger invocations
```

Or run the example script:

```bash
python examples/basic_solve.py
```

## How it works

```
User text → Parser → Formulator → CodeGen → Sandbox (subprocess)
                ↑                         ↓ failure
                └── Debugger (retry) ─────┘
                                    ↓ success
                              Interpreter → SolveResult
```

- **Raises** before execution for bad input or invalid agent output
  (`ClassificationError`, `FormulationError`, etc.)
- **Returns** `SolveResult` for solver outcomes (feasible, infeasible, retries exhausted)

See [docs/pipeline_design.md](docs/pipeline_design.md) for API semantics.

## Supported problem types (v0.1)

| Type | Solver | Notes |
|------|--------|-------|
| Scheduling | CP-SAT | Best tested |
| Allocation / knapsack | CP-SAT | Supported |
| Routing (VRP) | OR-Tools routing | Supported, less tested |

## Configuration

```python
harness = ORHarness(
    model="claude-sonnet-4-6",   # any LiteLLM-supported model
    max_retries=3,             # debugger attempts after failed execution
    timeout_seconds=30,        # sandbox subprocess timeout
    temperature=0.0,
)
```

## Development

```bash
pytest                    # unit tests (no API key required)
python evals/run_evals.py   # full benchmark (requires API key)
```

## License

Apache 2.0 — see [LICENSE](LICENSE).
