"""Tests for execution_utils helpers."""

import json

import pytest

from orharness.code_contract import RESULT_MARKER
from orharness.execution_utils import (
    execution_has_valid_result,
    execution_needs_debugger,
    is_infeasible_result,
    parse_solver_output,
)
from orharness.models import ExecutionResult


def _stdout(payload: dict, noise: str = "") -> str:
    body = json.dumps(payload)
    prefix = f"{noise}\n" if noise else ""
    return f"{prefix}{RESULT_MARKER}\n{body}"


def test_parse_solver_output():
    payload = {"feasible": True, "objective_value": 1.0}
    parsed = parse_solver_output(_stdout(payload))
    assert parsed["feasible"] is True


def test_parse_solver_output_with_noise_before_marker():
    payload = {"feasible": True, "objective_value": None}
    parsed = parse_solver_output(_stdout(payload, noise="solver starting"))
    assert parsed["feasible"] is True


def test_parse_solver_output_missing_marker_raises():
    with pytest.raises(ValueError, match="marker"):
        parse_solver_output('{"feasible": true}')


def test_execution_has_valid_result():
    good = ExecutionResult(success=True, raw_output=_stdout({"feasible": True}))
    bad = ExecutionResult(success=True, raw_output="no json here")
    assert execution_has_valid_result(good) is True
    assert execution_has_valid_result(bad) is False


def test_execution_needs_debugger():
    crash = ExecutionResult(success=False, error_message="exit 1")
    invalid = ExecutionResult(success=True, raw_output="garbage")
    valid = ExecutionResult(success=True, raw_output=_stdout({"feasible": True}))
    assert execution_needs_debugger(crash) is True
    assert execution_needs_debugger(invalid) is True
    assert execution_needs_debugger(valid) is False


def test_is_infeasible_result():
    assert is_infeasible_result({"feasible": False}) is True
    assert is_infeasible_result({"feasible": True}) is False
