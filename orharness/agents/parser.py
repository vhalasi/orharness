import json
from litellm import completion
from orharness.models import (
    ParsedProblem,
    ProblemType,
    Confidence,
    ObjectiveKind,
    ORHarnessConfig,
)
from orharness.exceptions import ClassificationError, LowConfidenceError
from orharness.json_utils import extract_json

PARSER_PROMPT = """You are an expert in operations research and mathematical optimization.

Your job is to analyze a problem description and determine:
1. Whether it is a solvable optimization problem
2. What type of optimization problem it is
3. Extract all relevant information needed to solve it
4. Whether the user wants feasibility only, or to minimize/maximize something

A valid optimization problem must have ALL three of these:
- Decision variables: things we are choosing or assigning
- Constraints: rules that limit the choices
- Objective: something to minimize or maximize, OR a feasibility goal

Classify the problem into exactly one of these types:
- scheduling: assigning people or tasks to time slots
- routing: finding optimal paths or sequences for vehicles or deliveries
- allocation: distributing limited resources across items or activities
- unknown: not a valid optimization problem, or cannot be determined

Return ONLY a valid JSON object with exactly this structure, no explanation, no markdown:
{
    "is_or_problem": true or false,
    "problem_type": "scheduling" or "routing" or "allocation" or "unknown",
    "confidence": "high" or "medium" or "low",
    "reason": "one sentence explaining your classification",
    "entities": {"key": value},
    "constraints": ["constraint 1", "constraint 2"],
    "objective_kind": "feasibility" or "minimize" or "maximize",
    "objective_description": "what to optimize, or null if feasibility"
}

Rules for objective_kind:
- Use "feasibility" when the user only states constraints and wants any valid solution.
  Set objective_description to null. Do NOT invent an objective.
- Use "minimize" when the user explicitly wants to minimize something (cost, distance,
  understaffing, hours, etc.). Set objective_description to a short phrase.
- Use "maximize" when the user explicitly wants to maximize something (value, profit,
  coverage, etc.). Set objective_description to a short phrase.

If is_or_problem is false, set problem_type to "unknown", objective_kind to "feasibility",
objective_description to null, and leave entities and constraints empty.
"""

def _validate_objective(parsed: ParsedProblem) -> None:
    if not parsed.is_or_problem:
        return

    if parsed.objective_kind == ObjectiveKind.FEASIBILITY:
        if parsed.objective_description:
            raise ClassificationError(
                "Feasibility problems must not have an objective_description"
            )
        return

    if not parsed.objective_description:
        raise ClassificationError(
            f"Optimization problems ({parsed.objective_kind.value}) "
            "must include objective_description"
        )


def parse_problem(user_input: str, config: ORHarnessConfig) -> ParsedProblem:
    response = completion(
        model=config.model,
        temperature=config.temperature,
        messages=[
            {"role": "system", "content": PARSER_PROMPT},
            {"role": "user", "content": user_input}
        ]
    )
    raw_text = response.choices[0].message.content

    try:
        data = json.loads(extract_json(raw_text))
    except json.JSONDecodeError:
        raise ClassificationError(
            f"Parser returned invalid JSON: {raw_text}"
        )

    try:
        parsed = ParsedProblem(**data)
    except Exception as e:
        raise ClassificationError(
            f"Parser returned invalid structure: {e}"
        )

    if not parsed.is_or_problem:
        raise ClassificationError(
            f"Not a valid optimization problem: {parsed.reason}"
        )

    if parsed.confidence == Confidence.LOW:
        raise LowConfidenceError(
            f"Classification confidence too low: {parsed.reason}"
        )

    _validate_objective(parsed)

    return parsed
