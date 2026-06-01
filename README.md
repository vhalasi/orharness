# ORHarness

> The missing bridge between large language models and operations research solvers.

**ORHarness** is an open-source Python library that lets any LLM solve real-world
optimization problems by connecting it to [Google OR-Tools](https://developers.google.com/optimization)
through a multi-agent pipeline. Describe your problem in plain English — ORHarness does the rest.

## Status

Early development. This is a scaffold; APIs and behavior are not yet implemented.

## Installation

```bash
pip install orharness
```

## Quickstart

```python
from orharness import ORHarness

harness = ORHarness(model="claude-sonnet-4-6")
result = harness.solve("I have 6 nurses, 3 shifts per day, max 40hrs/week...")

print(result.solution)            # plain English schedule
print(result.code)                # generated OR-Tools code
print(result.feasible)            # True / False
print(result.objective_value)     # numerical result
```

## License

Apache 2.0 — see [LICENSE](LICENSE).
