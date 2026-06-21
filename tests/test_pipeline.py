"""Tests for pipeline orchestration (mocked agents, no LLM calls)."""

import json
from unittest.mock import patch

import pytest

from orharness.code_contract import RESULT_MARKER
from orharness.exceptions import InterpretationError
from orharness.models import (
    Confidence,
    FormulatedModel,
    GeneratedCode,
    ObjectiveKind,
    ORHarnessConfig,
    ORSolver,
    ParsedProblem,
    ProblemType,
)
from orharness.pipeline import solve_problem


def _parsed(objective_kind: ObjectiveKind = ObjectiveKind.FEASIBILITY) -> ParsedProblem:
    return ParsedProblem(
        is_or_problem=True,
        problem_type=ProblemType.SCHEDULING,
        confidence=Confidence.HIGH,
        reason="test fixture",
        entities={"nurses": 4},
        constraints=["max 40 hours"],
        objective_kind=objective_kind,
        objective_description=None if objective_kind == ObjectiveKind.FEASIBILITY else "minimize cost",
    )


def _formulated(objective_kind: ObjectiveKind = ObjectiveKind.FEASIBILITY) -> FormulatedModel:
    return FormulatedModel(
        problem_type=ProblemType.SCHEDULING,
        variables=["x[i,j] binary"],
        objective_kind=objective_kind,
        objective=None if objective_kind == ObjectiveKind.FEASIBILITY else "minimize cost",
        constraints=["each shift covered"],
        parameters={"nurses": 4},
    )


def _generated(code: str = "print('stub')", attempt: int = 1) -> GeneratedCode:
    return GeneratedCode(
        code=code,
        solver=ORSolver.CP_SAT,
        problem_type=ProblemType.SCHEDULING,
        attempt=attempt,
    )


def _solver_stdout(feasible: bool = True, objective_value=None) -> str:
    payload = {
        "status": "OPTIMAL" if feasible else "INFEASIBLE",
        "feasible": feasible,
        "objective_value": objective_value,
        "solution": {"assignments": []},
        "solve_time_seconds": 0.05,
    }
    return f"{RESULT_MARKER}\n{json.dumps(payload)}"


@pytest.fixture
def config():
    return ORHarnessConfig(max_retries=2, timeout_seconds=10)


@patch("orharness.pipeline.interpret_solution", return_value="Schedule looks good.")
@patch("orharness.pipeline.generate_code")
@patch("orharness.pipeline.formulate_problem")
@patch("orharness.pipeline.parse_problem")
def test_solve_problem_feasible_success(
    mock_parse, mock_formulate, mock_codegen, mock_interpret, config
):
    mock_parse.return_value = _parsed()
    mock_formulate.return_value = _formulated()
    mock_codegen.return_value = _generated()

    with patch("orharness.pipeline.run_code") as mock_run:
        from orharness.models import ExecutionResult

        mock_run.return_value = ExecutionResult(
            success=True,
            raw_output=_solver_stdout(feasible=True),
            solve_time_seconds=0.1,
        )
        result = solve_problem("assign nurses to shifts", config)

    assert result.success is True
    assert result.feasible is True
    assert result.solution == "Schedule looks good."
    assert result.objective_value is None
    assert result.retries == 0


@patch("orharness.pipeline.interpret_solution", return_value="No valid schedule exists.")
@patch("orharness.pipeline.generate_code")
@patch("orharness.pipeline.formulate_problem")
@patch("orharness.pipeline.parse_problem")
def test_solve_problem_infeasible_still_success(
    mock_parse, mock_formulate, mock_codegen, mock_interpret, config
):
    mock_parse.return_value = _parsed()
    mock_formulate.return_value = _formulated()
    mock_codegen.return_value = _generated()

    with patch("orharness.pipeline.run_code") as mock_run:
        from orharness.models import ExecutionResult

        mock_run.return_value = ExecutionResult(
            success=True,
            raw_output=_solver_stdout(feasible=False),
        )
        result = solve_problem("impossible constraints", config)

    assert result.success is True
    assert result.feasible is False
    assert result.solution == "No valid schedule exists."


@patch("orharness.pipeline.debug_code")
@patch("orharness.pipeline.generate_code")
@patch("orharness.pipeline.formulate_problem")
@patch("orharness.pipeline.parse_problem")
def test_solve_problem_retries_then_succeeds(
    mock_parse, mock_formulate, mock_codegen, mock_debug, config
):
    mock_parse.return_value = _parsed()
    mock_formulate.return_value = _formulated()
    mock_codegen.return_value = _generated(attempt=1)
    mock_debug.return_value = _generated(attempt=2)

    from orharness.models import ExecutionResult

    fail_exec = ExecutionResult(success=False, error_message="crash")
    ok_exec = ExecutionResult(success=True, raw_output=_solver_stdout())

    with patch("orharness.pipeline.run_code", side_effect=[fail_exec, ok_exec]):
        with patch(
            "orharness.pipeline.interpret_solution",
            return_value="Done.",
        ):
            result = solve_problem("assign nurses", config)

    assert result.success is True
    assert result.retries == 1
    mock_debug.assert_called_once()


@patch("orharness.pipeline.debug_code")
@patch("orharness.pipeline.generate_code")
@patch("orharness.pipeline.formulate_problem")
@patch("orharness.pipeline.parse_problem")
def test_solve_problem_max_retries_exhausted(
    mock_parse, mock_formulate, mock_codegen, mock_debug, config
):
    mock_parse.return_value = _parsed()
    mock_formulate.return_value = _formulated()
    mock_codegen.return_value = _generated()
    mock_debug.return_value = _generated(attempt=2)

    from orharness.models import ExecutionResult

    fail_exec = ExecutionResult(success=False, error_message="still broken")

    with patch("orharness.pipeline.run_code", return_value=fail_exec):
        result = solve_problem("assign nurses", config)

    assert result.success is False
    assert result.retries == config.max_retries
    assert "exhausted" in (result.error or "").lower()


@patch("orharness.pipeline.interpret_solution", side_effect=InterpretationError("empty"))
@patch("orharness.pipeline.generate_code")
@patch("orharness.pipeline.formulate_problem")
@patch("orharness.pipeline.parse_problem")
def test_solve_problem_interpreter_failure(
    mock_parse, mock_formulate, mock_codegen, mock_interpret, config
):
    mock_parse.return_value = _parsed()
    mock_formulate.return_value = _formulated()
    mock_codegen.return_value = _generated()

    with patch("orharness.pipeline.run_code") as mock_run:
        from orharness.models import ExecutionResult

        mock_run.return_value = ExecutionResult(
            success=True,
            raw_output=_solver_stdout(feasible=True),
        )
        result = solve_problem("assign nurses", config)

    assert result.success is False
    assert result.feasible is True
    assert "empty" in (result.error or "")


@patch("orharness.pipeline.interpret_solution", return_value="Optimized.")
@patch("orharness.pipeline.generate_code")
@patch("orharness.pipeline.formulate_problem")
@patch("orharness.pipeline.parse_problem")
def test_solve_problem_optimization_objective_value(
    mock_parse, mock_formulate, mock_codegen, mock_interpret, config
):
    mock_parse.return_value = _parsed(ObjectiveKind.MINIMIZE)
    mock_formulate.return_value = _formulated(ObjectiveKind.MINIMIZE)
    mock_codegen.return_value = _generated()

    with patch("orharness.pipeline.run_code") as mock_run:
        from orharness.models import ExecutionResult

        mock_run.return_value = ExecutionResult(
            success=True,
            raw_output=_solver_stdout(feasible=True, objective_value=42.0),
        )
        result = solve_problem("minimize cost", config)

    assert result.success is True
    assert result.objective_value == 42.0
