import ast
import json
from litellm import completion
from orharness.models import (
    FormulatedModel,
    GeneratedCode,
    ProblemType,
    ObjectiveKind,
    ORSolver,
    ORHarnessConfig,
)
from orharness.exceptions import CodeGenerationError
from orharness.json_utils import extract_code

# Printed by the generated script immediately before its JSON result, so the
# pipeline can locate the result even if the solver or libraries print noise.
RESULT_MARKER = "===ORHARNESS_RESULT==="

# v0.1 solver routing. SCHEDULING and ALLOCATION both use CP-SAT, so codegen
# only has to be fluent in two OR-Tools APIs (CP-SAT and the routing library).
SOLVER_FOR_PROBLEM_TYPE = {
    ProblemType.SCHEDULING: ORSolver.CP_SAT,
    ProblemType.ALLOCATION: ORSolver.CP_SAT,
    ProblemType.ROUTING: ORSolver.ROUTING,
}

OUTPUT_PROTOCOL = f"""
OUTPUT PROTOCOL (mandatory):
The script must solve the model and then, as its FINAL action, print exactly
one line containing the marker {RESULT_MARKER} followed by a single line of
JSON with EXACTLY these keys:

    {RESULT_MARKER}
    {{"status": "<solver status>", "feasible": <true|false>, "objective_value": <number or null>, "solution": <object>, "solve_time_seconds": <number>}}

Rules for the result:
- status: the solver status as a string (e.g. "OPTIMAL", "FEASIBLE", "INFEASIBLE").
- feasible: true only if the solver found a usable solution.
- objective_value: the optimized value, or null for pure-feasibility problems
  or when no solution was found.
- solve_time_seconds: the solver wall-clock time as a float.
- Always print the marker + JSON, even when the problem is INFEASIBLE
  (in that case feasible=false, objective_value=null, solution={{}}).
- Use json.dumps to print the JSON. Print nothing after it.
"""

CP_SAT_GUIDANCE = """
Use Google OR-Tools CP-SAT (from ortools.sat.python import cp_model).
- Build the model with cp_model.CpModel().
- Create variables with model.NewBoolVar / model.NewIntVar.
- Add constraints with model.Add(...).
- Set the objective with model.Minimize(...) or model.Maximize(...) when there
  is one; omit it for pure feasibility.
- Solve with solver = cp_model.CpSolver(); status = solver.Solve(model).
- feasible is true when status is cp_model.OPTIMAL or cp_model.FEASIBLE.
- Read solve time from solver.WallTime().
- For the "solution" object, list the active assignments in a readable form,
  e.g. {"assignments": [{"entity": 0, "slot": "night", "time": "day_0"}, ...]}.
"""

ROUTING_GUIDANCE = """
Use Google OR-Tools routing library (from ortools.constraint_solver import
pywrapcp, routing_enums_pb2).
- Build a RoutingIndexManager and RoutingModel.
- Register the distance/transit callback and set the arc cost evaluator.
- Add capacity/time-window dimensions only if the model requires them.
- Solve with a SearchParameters object; read the objective from solution.ObjectiveValue().
- feasible is true when the solver returns a non-null solution.
- For the "solution" object, list each route in a readable form,
  e.g. {"routes": [{"vehicle": 0, "stops": [0, 3, 5, 0], "distance": 120}, ...]}.
"""

FEASIBILITY_GUIDANCE = """
This is a FEASIBILITY-ONLY problem (objective_kind=feasibility):
- Do NOT call model.Minimize(...) or model.Maximize(...).
- Do NOT add penalty, slack, or understaffing variables.
- Only implement the hard constraints and find any feasible solution.
- In the result JSON, always set "objective_value": null.
"""

OPTIMIZATION_GUIDANCE = """
This is an OPTIMIZATION problem (objective_kind is minimize or maximize):
- Implement the objective exactly as described in the formulation.
- In the result JSON, set objective_value to the solver's optimized value.
"""

GUIDANCE_FOR_SOLVER = {
    ORSolver.CP_SAT: CP_SAT_GUIDANCE,
    ORSolver.ROUTING: ROUTING_GUIDANCE,
}


def _build_system_prompt(solver: ORSolver, objective_kind: ObjectiveKind) -> str:
    objective_guidance = (
        FEASIBILITY_GUIDANCE
        if objective_kind == ObjectiveKind.FEASIBILITY
        else OPTIMIZATION_GUIDANCE
    )
    return f"""You are an expert Google OR-Tools programmer.

You will receive a precise mathematical formulation of an optimization problem.
Your job is to write a COMPLETE, STANDALONE Python script that builds and solves
it with OR-Tools.

Requirements:
- The script must be fully self-contained and runnable with `python script.py`.
- Inline ALL parameters from the formulation as literals. Do not read from files,
  stdin, the network, or any external source.
- Translate every variable, constraint, and the objective faithfully and exactly.
{GUIDANCE_FOR_SOLVER[solver]}{objective_guidance}{OUTPUT_PROTOCOL}
Return ONLY the Python code. No explanation, no markdown fences.
"""


def _validate_generated_code(formulated: FormulatedModel, code: str) -> None:
    if formulated.objective_kind == ObjectiveKind.FEASIBILITY:
        if "Minimize" in code or "Maximize" in code:
            raise CodeGenerationError(
                "Feasibility problem must not set an objective in generated code"
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

    formulation_summary = f"""
Problem type: {formulated.problem_type.value}
Objective kind: {formulated.objective_kind.value}
Variables: {json.dumps(formulated.variables)}
Objective: {formulated.objective}
Constraints: {json.dumps(formulated.constraints)}
Parameters: {json.dumps(formulated.parameters)}
"""

    response = completion(
        model=config.model,
        temperature=config.temperature,
        messages=[
            {
                "role": "system",
                "content": _build_system_prompt(solver, formulated.objective_kind),
            },
            {"role": "user", "content": formulation_summary}
        ]
    )

    raw_text = response.choices[0].message.content
    code = extract_code(raw_text)

    if not code.strip():
        raise CodeGenerationError("Code generator returned empty code")

    try:
        ast.parse(code)
    except SyntaxError as e:
        raise CodeGenerationError(
            f"Code generator returned code with a syntax error: {e}"
        )

    if "ortools" not in code:
        raise CodeGenerationError(
            "Generated code does not import ortools"
        )

    _validate_generated_code(formulated, code)

    return GeneratedCode(
        code=code,
        solver=solver,
        problem_type=formulated.problem_type,
        attempt=1,
    )
