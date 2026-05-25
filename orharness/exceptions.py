class ORHarnessError(Exception):
    """Base exception for all ORHarness errors."""
    pass


class ClassificationError(ORHarnessError):
    """Raised when the parser cannot classify the problem."""
    pass


class LowConfidenceError(ORHarnessError):
    """Raised when classification confidence is too low to proceed."""
    pass


class FormulationError(ORHarnessError):
    """Raised when the formulator cannot produce a valid mathematical model."""
    pass


class CodeGenerationError(ORHarnessError):
    """Raised when the code generation agent fails to produce valid code."""
    pass


class SandboxTimeoutError(ORHarnessError):
    """Raised when the solver exceeds the timeout limit."""
    pass


class SolverInfeasibleError(ORHarnessError):
    """Raised when OR-Tools finds no feasible solution."""
    pass


class MaxRetriesExceededError(ORHarnessError):
    """Raised when the debugger agent exhausts all retry attempts."""
    pass