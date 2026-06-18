import json
from litellm import completion
from orharness.models import (
    ParsedProblem,
    FormulatedModel,
    ORHarnessConfig,
)
from orharness.exceptions import InterpretationError

INTERPRETER_PROMPT = """You are an expert operations research consultant.

Explain the solver result to the user in plain English. Be concise and specific.
Reference the original problem. Format assignments, routes, or allocations clearly
so a non-technical reader can act on the answer.

If feasible is false, state that no solution satisfies all constraints. Summarize
what the solver reported without inventing reasons the model did not provide.

Do not mention OR-Tools, Python, JSON, or internal solver mechanics unless helpful.
Return only the explanation text — no markdown headers, no code blocks."""


def interpret_solution(
    user_input: str,
    parsed: ParsedProblem,
    formulated: FormulatedModel,
    solver_result: dict,
    config: ORHarnessConfig,
) -> str:
    """Turn structured solver output into a plain-English answer."""

    user_message = f"""
Original problem:
{user_input}

Problem type: {parsed.problem_type.value}
Objective: {parsed.objective_kind.value}
{f"Objective description: {parsed.objective_description}" if parsed.objective_description else ""}

Solver result:
{json.dumps(solver_result, indent=2)}
"""

    response = completion(
        model=config.model,
        temperature=config.temperature,
        messages=[
            {"role": "system", "content": INTERPRETER_PROMPT},
            {"role": "user", "content": user_message},
        ],
    )

    raw_text = response.choices[0].message.content
    if not raw_text or not raw_text.strip():
        raise InterpretationError("Interpreter returned empty response")

    return raw_text.strip()
