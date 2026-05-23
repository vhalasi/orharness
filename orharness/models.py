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
    
    