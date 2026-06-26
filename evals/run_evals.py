"""Run ORHarness against benchmark problems (requires API key in .env)."""

import argparse
import json
import sys
import time
from pathlib import Path

DEFAULT_PROBLEMS = Path(__file__).parent / "problems" / "v0.1.jsonl"


def load_problems(path: Path) -> list[dict]:
    problems = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                problems.append(json.loads(line))
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description="Run ORHarness eval problems")
    parser.add_argument(
        "--problems",
        type=Path,
        default=DEFAULT_PROBLEMS,
        help="Path to JSONL problem file",
    )
    parser.add_argument("--model", default="claude-sonnet-4-6")
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument(
        "--output",
        type=Path,
        help="Write JSONL results to this file",
    )
    args = parser.parse_args()

    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    from orharness import ORHarness

    problems = load_problems(args.problems)
    harness = ORHarness(
        model=args.model,
        max_retries=args.max_retries,
        timeout_seconds=args.timeout,
    )

    results = []
    success_count = 0
    feasible_count = 0
    start = time.monotonic()

    for i, problem in enumerate(problems, 1):
        pid = problem["id"]
        text = problem["input"]
        expect_feasible = problem.get("expect_feasible", True)
        print(f"[{i}/{len(problems)}] {pid} ...", flush=True)

        row = {
            "id": pid,
            "problem_type": problem.get("problem_type"),
            "expect_feasible": expect_feasible,
        }
        try:
            t0 = time.monotonic()
            result = harness.solve(text)
            elapsed = time.monotonic() - t0
            row.update(
                {
                    "success": result.success,
                    "feasible": result.feasible,
                    "retries": result.retries,
                    "objective_value": result.objective_value,
                    "solve_time_seconds": result.solve_time_seconds,
                    "elapsed_seconds": round(elapsed, 2),
                    "error": result.error,
                }
            )
            if result.success:
                success_count += 1
            if result.feasible:
                feasible_count += 1
            status = "OK" if result.success and result.feasible else (
                "infeasible" if result.success and not result.feasible else "FAIL"
            )
            print(f"  → {status} (retries={result.retries}, {elapsed:.1f}s)")
        except Exception as exc:
            row.update({"success": False, "feasible": False, "error": str(exc)})
            print(f"  → ERROR: {exc}")

        results.append(row)

    total_elapsed = time.monotonic() - start
    n = len(problems)
    print()
    print(f"Completed {n} problems in {total_elapsed:.1f}s")
    print(f"  pipeline success: {success_count}/{n} ({100 * success_count / n:.0f}%)")
    print(f"  feasible:         {feasible_count}/{n} ({100 * feasible_count / n:.0f}%)")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as fh:
            for row in results:
                fh.write(json.dumps(row) + "\n")
        print(f"Results written to {args.output}")

    return 0 if success_count == n else 1


if __name__ == "__main__":
    sys.exit(main())
