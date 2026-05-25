import json
from litellm import completion
from orharness.models import ParsedProblem, ProblemType, Confidence, ORHarnessConfig
from orharness.exceptions import ClassificationError, LowConfidenceError

PARSER_PROMPT = """You are an expert in operations research and mathematical optimization.

Your job is to analyze a problem description and determine:
1. Whether it is a solvable optimization problem
2. What type of optimization problem it is
3. Extract all relevant information needed to solve it

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
    "objective": "what to optimize or null if just feasibility"
}

If is_or_problem is false, set problem_type to "unknown" and leave entities, constraints, objective empty.
"""

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
        data = json.loads(raw_text)
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

    return parsed