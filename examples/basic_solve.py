"""Minimal ORHarness example — nurse scheduling."""

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from orharness import ORHarness

PROBLEM = """
Six nurses must cover three shifts per day (morning, afternoon, night) for one week.
Each nurse works at most 40 hours per week. At least two nurses must be on each shift.
Find any valid schedule.
"""

if __name__ == "__main__":
    harness = ORHarness()
    result = harness.solve(PROBLEM.strip())

    print(f"success={result.success} feasible={result.feasible} retries={result.retries}")
    if result.error:
        print(f"error: {result.error}")
    if result.solution:
        print()
        print(result.solution)
