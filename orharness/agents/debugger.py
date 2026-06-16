from litellm import completion
from orharness.models import (
    FormulatedModel,
    GeneratedCode,
    ExecutionResult,
    ORHarnessConfig,
)
from orharness.code_contract import (
    format_formulation_summary,
    build_debug_system_prompt,
    validate_llm_code,
)

_ERROR_TAIL = 2000
_STDOUT_TAIL = 1500


def _format_failure_context(execution: ExecutionResult) -> str:
    if execution.error_message:
        error_text = execution.error_message[-_ERROR_TAIL:]
    elif execution.success:
        error_text = "Script exited successfully but did not print a valid result."
    else:
        error_text = "Script failed with no stderr."

    stdout_text = (execution.raw_output or "")[-_STDOUT_TAIL:]
    return f"""
Error:
{error_text}

Stdout (tail):
{stdout_text if stdout_text else "(empty)"}
"""


def debug_code(
    generated: GeneratedCode,
    execution: ExecutionResult,
    formulated: FormulatedModel,
    config: ORHarnessConfig,
) -> GeneratedCode:
    """Rewrite failed generated code using the execution error and formulation."""

    user_message = f"""
Attempt: {generated.attempt + 1} (retry after failed execution)

{format_formulation_summary(formulated)}

Broken code:
{generated.code}

{_format_failure_context(execution)}
"""

    response = completion(
        model=config.model,
        temperature=config.temperature,
        messages=[
            {
                "role": "system",
                "content": build_debug_system_prompt(
                    generated.solver, formulated.objective_kind
                ),
            },
            {"role": "user", "content": user_message},
        ],
    )

    raw_text = response.choices[0].message.content
    code = validate_llm_code(raw_text, formulated)

    return GeneratedCode(
        code=code,
        solver=generated.solver,
        problem_type=generated.problem_type,
        attempt=generated.attempt + 1,
    )
