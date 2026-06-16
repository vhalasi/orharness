"""Helpers for interpreting sandbox output (used by pipeline and tests)."""

import json

from orharness.code_contract import RESULT_MARKER
from orharness.models import ExecutionResult


def parse_solver_output(raw_output: str) -> dict:
    """Extract and parse the JSON result printed after RESULT_MARKER."""
    if RESULT_MARKER not in raw_output:
        raise ValueError("Result marker not found in stdout")

    payload = raw_output.split(RESULT_MARKER, 1)[1].strip()
    return json.loads(payload)


def execution_has_valid_result(execution: ExecutionResult) -> bool:
    """True when stdout contains a parseable solver result JSON."""
    if not execution.raw_output:
        return False
    try:
        parse_solver_output(execution.raw_output)
    except (ValueError, json.JSONDecodeError):
        return False
    return True


def execution_needs_debugger(execution: ExecutionResult) -> bool:
    """True when execution failed or did not produce a valid result payload."""
    if not execution.success:
        return True
    return not execution_has_valid_result(execution)


def is_infeasible_result(parsed: dict) -> bool:
    """True when the solver ran correctly but found no usable solution."""
    return not parsed.get("feasible", False)
