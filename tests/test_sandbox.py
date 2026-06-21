"""Tests for sandbox execution."""

import json
import time

import pytest

from orharness.code_contract import RESULT_MARKER
from orharness.exceptions import SandboxTimeoutError
from orharness.models import GeneratedCode, ORHarnessConfig, ORSolver, ProblemType
from orharness.sandbox import run_code


def _generated(code: str) -> GeneratedCode:
    return GeneratedCode(
        code=code,
        solver=ORSolver.CP_SAT,
        problem_type=ProblemType.SCHEDULING,
    )


def test_run_code_success_with_result_marker():
    payload = {
        "status": "OPTIMAL",
        "feasible": True,
        "objective_value": None,
        "solution": {"assignments": []},
        "solve_time_seconds": 0.01,
    }
    code = (
        "import json\n"
        f"payload = {repr(payload)}\n"
        f"print('{RESULT_MARKER}')\n"
        "print(json.dumps(payload))\n"
    )
    config = ORHarnessConfig(timeout_seconds=10)
    result = run_code(_generated(code), config)

    assert result.success is True
    assert RESULT_MARKER in (result.raw_output or "")
    assert result.solve_time_seconds is not None


def test_run_code_crash_nonzero_exit():
    code = "raise RuntimeError('boom')"
    config = ORHarnessConfig(timeout_seconds=10)
    result = run_code(_generated(code), config)

    assert result.success is False
    assert result.error_message is not None


def test_run_code_timeout():
    code = "import time\nwhile True: time.sleep(60)"
    config = ORHarnessConfig(timeout_seconds=1)
    with pytest.raises(SandboxTimeoutError):
        run_code(_generated(code), config)
