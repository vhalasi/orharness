"""End-to-end orchestration: parse → formulate → codegen → sandbox → debugger → interpret."""

from orharness.agents.codegen import generate_code
from orharness.agents.debugger import debug_code
from orharness.agents.formulator import formulate_problem
from orharness.agents.interpreter import interpret_solution
from orharness.agents.parser import parse_problem
from orharness.exceptions import InterpretationError, SandboxTimeoutError
from orharness.execution_utils import (
    execution_has_valid_result,
    is_infeasible_result,
    parse_solver_output,
)
from orharness.models import (
    ExecutionResult,
    ObjectiveKind,
    ORHarnessConfig,
    SolveResult,
)
from orharness.sandbox import run_code


def _objective_value_from_result(
    parsed_objective_kind: ObjectiveKind,
    solver_result: dict,
) -> float | None:
    if parsed_objective_kind == ObjectiveKind.FEASIBILITY:
        return None

    raw = solver_result.get("objective_value")
    if raw is None:
        return None
    return float(raw)


def _solve_time_from_result(
    solver_result: dict,
    execution: ExecutionResult,
) -> float | None:
    raw = solver_result.get("solve_time_seconds")
    if raw is not None:
        return float(raw)
    return execution.solve_time_seconds


def solve_problem(user_input: str, config: ORHarnessConfig) -> SolveResult:
    """Run the full ORHarness pipeline and return a SolveResult.

    Raises ClassificationError, LowConfidenceError, FormulationError, or
    CodeGenerationError when the pipeline cannot start execution.

    Returns SolveResult for execution outcomes (feasible, infeasible, retries
    exhausted, or interpreter failure).
    """
    parsed = parse_problem(user_input, config)
    formulated = formulate_problem(parsed, config)
    generated = generate_code(formulated, config)

    retries = 0
    execution: ExecutionResult | None = None
    solver_result: dict | None = None

    while True:
        try:
            execution = run_code(generated, config)
        except SandboxTimeoutError as exc:
            execution = ExecutionResult(
                success=False,
                error_message=str(exc),
            )

        if execution_has_valid_result(execution):
            solver_result = parse_solver_output(execution.raw_output or "")
            break

        if retries >= config.max_retries:
            detail = execution.error_message or "Script did not print a valid result."
            return SolveResult(
                success=False,
                code=generated.code,
                retries=retries,
                error=(
                    f"Debugger exhausted {config.max_retries} retries: {detail}"
                ),
            )

        generated = debug_code(generated, execution, formulated, config)
        retries += 1

    feasible = not is_infeasible_result(solver_result)
    objective_value = _objective_value_from_result(
        parsed.objective_kind, solver_result
    )
    solve_time_seconds = _solve_time_from_result(solver_result, execution)

    try:
        solution_text = interpret_solution(
            user_input,
            parsed,
            formulated,
            solver_result,
            config,
        )
    except InterpretationError as exc:
        return SolveResult(
            success=False,
            code=generated.code,
            feasible=feasible,
            objective_value=objective_value,
            retries=retries,
            solve_time_seconds=solve_time_seconds,
            error=str(exc),
        )

    return SolveResult(
        success=True,
        solution=solution_text,
        code=generated.code,
        feasible=feasible,
        objective_value=objective_value,
        retries=retries,
        solve_time_seconds=solve_time_seconds,
    )
