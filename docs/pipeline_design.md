# ORHarness Pipeline Design

This document records the architecture and API decisions for `pipeline.py`,
the interpreter agent, and the public `ORHarness.solve()` entry point.

## Overview

```
User text
  → parser          (raises on bad input / low confidence)
  → formulator      (raises on invalid formulation)
  → codegen         (raises on invalid code structure)
  → sandbox         (subprocess, no LLM)
  → debugger loop   (on crash, timeout, missing/invalid JSON)
  → interpreter     (plain English)
  → SolveResult
```

The pipeline owns the debugger retry loop. Individual agents are single-shot
functions; only `solve_problem()` orchestrates retries.

## Public API

```python
from orharness import ORHarness

harness = ORHarness(model="claude-sonnet-4-6")
result = harness.solve("I have 6 nurses, 3 shifts per day...")

print(result.success)           # harness completed end-to-end?
print(result.feasible)          # solver found a usable solution?
print(result.solution)          # plain English (feasible or infeasible)
print(result.code)              # final generated script
print(result.objective_value)   # None for feasibility-only problems
print(result.retries)           # debugger invocations
print(result.solve_time_seconds)
print(result.error)             # set when success=False
```

`ORHarnessConfig` can also be built directly and passed to `solve_problem()`
for programmatic use without the `ORHarness` wrapper.

## Error handling: raise vs return

Two categories of failure:

| Category | Examples | Handling |
|----------|----------|----------|
| **Pre-execution** | Not an OR problem, low confidence, bad formulation, codegen validation failed | **Raise** exception |
| **Execution outcome** | Infeasible, retries exhausted, interpreter failed | **Return** `SolveResult` |

### Exceptions (raised before sandbox)

| Exception | When |
|-----------|------|
| `ClassificationError` | Parser rejects input (not OR, invalid JSON/structure) |
| `LowConfidenceError` | Parser confidence is `low` |
| `FormulationError` | Formulator output invalid or inconsistent with parser |
| `CodeGenerationError` | Codegen output invalid (AST, missing ortools, wrong solver mapping) |

### SolveResult (returned after sandbox starts)

| Situation | `success` | `feasible` | `error` |
|-----------|-----------|------------|---------|
| Feasible solution | `True` | `True` | `None` |
| Infeasible (valid solver JSON) | `True` | `False` | `None` |
| Debugger retries exhausted | `False` | `False` | message with retry count |
| Interpreter failed | `False` | prior value | interpreter message |

`SolverInfeasibleError` and `MaxRetriesExceededError` remain defined for future
`strict=True` mode but are **not raised** by `solve()` in v0.1.

## Field semantics

### `success`

`True` when the pipeline finished cleanly: valid sandbox JSON was parsed and
the interpreter produced text. Includes **infeasible** problems — the solver
gave a definitive answer and the user gets an explanation.

`False` when the harness failed: debugger exhausted retries or the interpreter
could not produce a response.

### `feasible`

Mathematical outcome from solver JSON (`feasible` key). Only meaningful when
the sandbox produced valid result JSON. Defaults to `False` on early failure paths.

### `objective_value`

Taken from solver JSON `objective_value`, coerced to `float`.

**Forced to `None`** when `parsed.objective_kind == feasibility`, even if the
generated script incorrectly prints a number. Defense in depth against codegen
mistakes on feasibility-only problems.

### `retries`

Count of **debugger invocations** (not total sandbox runs). First codegen attempt
is not counted. With `max_retries=3`, up to 4 sandbox runs and 3 debugger calls.

### `solve_time_seconds`

Prefer `solve_time_seconds` from solver JSON (solver wall time). Fall back to
sandbox subprocess elapsed time if the JSON omits it.

## Debugger retry rules

Retry when:

- Process crash (non-zero exit)
- `SandboxTimeoutError` (caught in pipeline, passed to debugger as failed execution)
- Missing `===ORHARNESS_RESULT===` marker
- Invalid JSON after marker

Do **not** retry when:

- Valid JSON with `feasible: false` → pipeline continues to interpreter

The sandbox does not parse JSON. Routing logic lives in `execution_utils.py`
and the pipeline.

## Interpreter

`interpret_solution(user_input, parsed, formulated, solver_result, config) → str`

Single LiteLLM call. Input includes the original question and full solver JSON.
Output is plain English — no new Pydantic model.

For infeasible results, the interpreter explains that no solution exists rather
than raising an exception.

Raises `InterpretationError` on empty LLM response; the pipeline catches this
and returns `SolveResult(success=False, error=...)`.

## File layout

| File | Responsibility |
|------|----------------|
| `orharness/pipeline.py` | `solve_problem()` orchestration |
| `orharness/agents/interpreter.py` | `interpret_solution()` |
| `orharness/__init__.py` | `ORHarness` class and public exports |
| `orharness/execution_utils.py` | Sandbox output parsing and routing helpers |
| `orharness/sandbox.py` | Subprocess execution only |

## Future extensions (not v0.1)

- `strict=True` on `ORHarness`: raise `SolverInfeasibleError` / `MaxRetriesExceededError` instead of returning
- Streaming progress callbacks per pipeline stage
- Re-formulation on repeated debugger failure
- MCP server wrapping `solve()`
