import json
from litellm import completion
from orharness.models import (
    ParsedProblem,
    FormulatedModel,
    ProblemType,
    ORHarnessConfig
)
from orharness.exceptions import FormulationError
from orharness.json_utils import extract_json

FORMULATOR_PROMPT = """You are an expert operations research mathematician.

You will receive a parsed optimization problem and your job is to produce
a precise mathematical formulation that a programmer can directly translate
into OR-Tools code.

Be precise and explicit. Use exact variable names, exact numbers, exact constraint descriptions.
The output will be used directly to generate solver code — vagueness will cause errors.

For scheduling problems use binary decision variables: x[i][j][k] = 1 if entity i is assigned to slot j on day k.
For routing problems define distance matrices, vehicle capacities, and node lists explicitly.
For allocation problems define item values, weights, and capacity limits explicitly.

Return ONLY a valid JSON object with exactly this structure, no explanation, no markdown:
{
    "problem_type": "scheduling" or "routing" or "allocation",
    "variables": [
        "description of each decision variable with its type and indices"
    ],
    "objective": "precise mathematical description of what to minimize or maximize",
    "constraints": [
        "precise formal description of each constraint with exact numbers"
    ],
    "parameters": {
        "key": value
    }
}

Rules:
- variables must be explicit: include type (binary/integer/continuous), name, and indices
- constraints must include exact numbers from the problem, not vague descriptions
- parameters must contain ALL numerical values needed to build the model
- if the problem only requires feasibility, set objective to "find feasible solution"
"""

def formulate_problem(
    parsed: ParsedProblem,
    config: ORHarnessConfig
) -> FormulatedModel:

    problem_summary = f"""
Problem type: {parsed.problem_type.value}
Entities: {json.dumps(parsed.entities)}
Constraints: {json.dumps(parsed.constraints)}
Objective: {parsed.objective}
"""

    response = completion(
        model=config.model,
        temperature=config.temperature,
        messages=[
            {"role": "system", "content": FORMULATOR_PROMPT},
            {"role": "user", "content": problem_summary}
        ]
    )

    raw_text = response.choices[0].message.content

    try:
        data = json.loads(extract_json(raw_text))
    except json.JSONDecodeError:
        raise FormulationError(
            f"Formulator returned invalid JSON: {raw_text}"
        )

    try:
        formulated = FormulatedModel(**data)
    except Exception as e:
        raise FormulationError(
            f"Formulator returned invalid structure: {e}"
        )

    if not formulated.variables:
        raise FormulationError(
            "Formulator returned no decision variables"
        )

    if not formulated.constraints:
        raise FormulationError(
            "Formulator returned no constraints"
        )

    return formulated