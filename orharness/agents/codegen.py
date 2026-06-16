import json
from litellm import completion
from orharness.models import (
    FormulatedModel,
    GeneratedCode,
    ORHarnessConfig,
)
from orharness.exceptions import CodeGenerationError
from orharness.code_contract import (
    RESULT_MARKER,
    SOLVER_FOR_PROBLEM_TYPE,
    format_formulation_summary,
    build_codegen_system_prompt,
    validate_llm_code,
)

def generate_code(
    formulated: FormulatedModel,
    config: ORHarnessConfig
) -> GeneratedCode:

    solver = SOLVER_FOR_PROBLEM_TYPE.get(formulated.problem_type)
    if solver is None:
        raise CodeGenerationError(
            f"No solver mapping for problem type: {formulated.problem_type}"
        )

    response = completion(
        model=config.model,
        temperature=config.temperature,
        messages=[
            {
                "role": "system",
                "content": build_codegen_system_prompt(
                    solver, formulated.objective_kind
                ),
            },
            {"role": "user", "content": format_formulation_summary(formulated)},
        ]
    )

    raw_text = response.choices[0].message.content
    code = validate_llm_code(raw_text, formulated)

    return GeneratedCode(
        code=code,
        solver=solver,
        problem_type=formulated.problem_type,
        attempt=1,
    )
