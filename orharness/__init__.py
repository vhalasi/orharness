from orharness.exceptions import (
    ClassificationError,
    CodeGenerationError,
    FormulationError,
    InterpretationError,
    LowConfidenceError,
    MaxRetriesExceededError,
    ORHarnessError,
    SandboxTimeoutError,
    SolverInfeasibleError,
)
from orharness.models import ORHarnessConfig, SolveResult
from orharness.pipeline import solve_problem


class ORHarness:
    """LLM-powered harness for solving optimization problems with OR-Tools."""

    def __init__(
        self,
        model: str = "claude-sonnet-4-6",
        max_retries: int = 3,
        timeout_seconds: int = 30,
        temperature: float = 0.0,
    ):
        self.config = ORHarnessConfig(
            model=model,
            max_retries=max_retries,
            timeout_seconds=timeout_seconds,
            temperature=temperature,
        )

    def solve(self, user_input: str) -> SolveResult:
        return solve_problem(user_input, self.config)


__all__ = [
    "ORHarness",
    "ORHarnessConfig",
    "ORHarnessError",
    "SolveResult",
    "ClassificationError",
    "LowConfidenceError",
    "FormulationError",
    "CodeGenerationError",
    "InterpretationError",
    "SandboxTimeoutError",
    "SolverInfeasibleError",
    "MaxRetriesExceededError",
    "solve_problem",
]
