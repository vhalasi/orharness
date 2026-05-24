from enum import Enum
from typing import Optional
from pydantic import BaseModel, field_validator

class ProblemType(str, Enum):
    SCHEDULING = "scheduling"
    ROUTING = "routing"
    ALLOCATION = "allocation"
    UNKNOWN = "unknown"
    
class Confidence(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

class ParsedProblem(BaseModel):
    is_or_problem: bool
    problem_type: ProblemType
    confidence: Confidence
    reason: str
    entities: dict
    constraints: list[str]
    objective: Optional[str] = None
    
class FormulatedModel(BaseModel):
    problem_type: ProblemType
    variables: list[str]
    objective: str
    constraints: list[str]
    parameters: dict
    
class ORSolver(str, Enum):
    CP_SAT = "cp_sat"
    ROUTING = "routing"
    GLOP = "glop"        # for LINEAR in v0.2
    MIP = "mip"          # for INTEGER in v0.2
class GeneratedCode(BaseModel):
    code: str
    solver: ORSolver
    problem_type: ProblemType
    attempt: int = 1
    
class ExecutionResult(BaseModel):
    success: bool
    raw_output: Optional[str] = None
    error_message: Optional[str] = None
    solve_time_seconds: Optional[float] = None
    
class SolveResult(BaseModel):
    success: bool
    solution: Optional[str] = None
    code: Optional[str] = None
    feasible: bool = False
    objective_value: Optional[float] = None
    retries: int = 0
    solve_time_seconds: Optional[float] = None
    error: Optional[str] = None

class ORHarnessConfig(BaseModel):
    model: str = "claude-sonnet-4-20250514"
    max_retries: int = 3
    timeout_seconds: int = 30
    temperature: float = 0.0  
    