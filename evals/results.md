# Eval results

Run the v0.1 benchmark locally (requires `ANTHROPIC_API_KEY` in `.env`):

```bash
pip install -e ".[dev]"
python evals/run_evals.py --output evals/results.jsonl
```

## v0.1 problem set

`evals/problems/v0.1.jsonl` — 8 problems:

- 6 scheduling (5 feasibility, 1 intentionally infeasible)
- 1 vehicle routing
- 1 knapsack / allocation

## Metrics to track

- **Pipeline success** — `result.success` (solver ran and interpreter returned)
- **Feasible** — `result.feasible` (solver found a usable solution)
- **Retries** — average debugger invocations
- **Time** — `elapsed_seconds` per problem

Publish summary numbers here after a full run.
