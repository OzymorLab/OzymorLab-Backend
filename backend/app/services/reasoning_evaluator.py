"""
Reasoning Evaluator — Structured Reasoning Pipeline.

Handles: stepwise logic, derivation flow, mathematical proofs,
presentation quality, and procedural correctness.

Extends the SymPy validator with LLM-based reasoning evaluation:
  - For derivation steps: SymPy validates math → LLM evaluates logic flow.
  - For presentation steps: LLM evaluates structure and ordering.
"""
import logging

from app.services.llm_client import (
    call_llm, parse_json_response, get_grading_system_prompt,
)
from app.services.sympy_validator import validate_expected_against_student

logger = logging.getLogger(__name__)

REASONING_SYSTEM_PROMPT_SUFFIX = """

ADDITIONAL INSTRUCTIONS FOR REASONING EVALUATION:
You are evaluating the REASONING QUALITY and LOGICAL FLOW of a student's answer.
Focus on:
- Whether steps follow logically from one another
- Whether the derivation/proof procedure is correct
- Whether presentation is clear and well-structured
- Whether the student shows understanding of the method, not just the answer

Award marks for correct methodology even if the final answer contains minor errors.
Penalize steps that skip crucial logical connections or use incorrect formulas."""


def evaluate_reasoning_component(
    component: dict,
    rubric_steps: list[dict],
    student_steps: list[dict],
    board_notes: str = "",
    temperature: float = 0.0,
    subject: str = "General",
    board: str = "Generic",
    grade_level: str = "Unknown",
    api_key: str | None = None,
) -> dict:
    """
    Evaluate a reasoning-type component.

    Args:
        component: Decomposed component dict (type="reasoning").
        rubric_steps: Full rubric steps list.
        student_steps: Parsed student answer steps.

    Returns:
        Component evaluation result dict.
    """
    component_step_nums = set(component.get("rubric_steps", []))
    relevant_rubric = [
        s for s in rubric_steps if s.get("step_num") in component_step_nums
    ]

    if not relevant_rubric:
        return {
            "type": "reasoning",
            "marks_awarded": 0,
            "max_marks": component.get("max_marks", 0),
            "confidence": 0.0,
            "grade_distribution": [1.0],
            "step_grades": [],
            "justification": "No rubric steps for reasoning component.",
        }

    student_steps_by_num = {
        s.get("step_num", i + 1): s for i, s in enumerate(student_steps)
    }

    # For reasoning, we evaluate the derivation flow as a whole
    # First: SymPy validation on all derivation equations
    sympy_results = {}
    for rubric_step in relevant_rubric:
        if rubric_step.get("expected_exprs"):
            # Find matching student equations for THIS step only
            step_num = rubric_step.get("step_num")
            student_step = student_steps_by_num.get(step_num)
            student_eqs = student_step.get("equations", []) if student_step else []

            if student_eqs:
                validations = validate_expected_against_student(
                    rubric_step["expected_exprs"], student_eqs
                )
                all_valid = all(v.get("valid") is True for v in validations)
                any_invalid = any(v.get("valid") is False for v in validations)
                if all_valid:
                    sympy_results[rubric_step.get("step_num")] = {
                        "valid": True, "error": None
                    }
                elif any_invalid:
                    errors = [
                        v["error"] for v in validations if v.get("valid") is False
                    ]
                    sympy_results[rubric_step.get("step_num")] = {
                        "valid": False, "error": "; ".join(errors)
                    }
                else:
                    sympy_results[rubric_step.get("step_num")] = {
                        "valid": None, "error": "Could not parse", "fallback": True
                    }

    # Then: LLM evaluation of the full reasoning chain
    step_grades = []
    for rubric_step in relevant_rubric:
        rs_num = rubric_step.get("step_num", 0)
        max_marks = rubric_step.get("marks", 1)

        # Find best matching student step (simple positional fallback)
        student_step = student_steps_by_num.get(rs_num)
        if not student_step and student_steps:
            # Try to find any step with equations
            for s in student_steps:
                if s.get("equations") or s.get("step_type") in ("derivation", "result"):
                    student_step = s
                    break

        sympy_result = sympy_results.get(rs_num)

        if student_step is None:
            dist = [0.0] * (max_marks + 1)
            dist[0] = 1.0
            step_grades.append({
                "step_num": rs_num,
                "marks_awarded": 0,
                "max_marks": max_marks,
                "grade_distribution": dist,
                "justification": "No student reasoning found for this step.",
                "error_type": "missing_step",
                "sympy_valid": None,
            })
            continue

        # Build reasoning-specific prompt
        sympy_ctx = ""
        if sympy_result:
            if sympy_result.get("valid") is True:
                sympy_ctx = "[SYMBOLIC VALIDATION: Mathematical transformation is CORRECT]"
            elif sympy_result.get("valid") is False:
                sympy_ctx = f"[SYMBOLIC VALIDATION: ERROR — {sympy_result.get('error', 'Unknown')}]"

        prompt = f"""RUBRIC STEP {rs_num} (max {max_marks} marks — REASONING/DERIVATION):
Description: {rubric_step.get('description', '')}
Marking notes: {rubric_step.get('marking_notes', '')}
Expected expressions: {rubric_step.get('expected_exprs', [])}

STUDENT REASONING:
{student_step.get('text', '')}
Student equations: {student_step.get('equations', [])}

{sympy_ctx}

Evaluate the REASONING QUALITY. Consider:
1. Logical flow from premise to conclusion
2. Mathematical correctness of each transformation
3. Completeness of the derivation
4. Presentation clarity

Return JSON:
{{"marks_awarded": <int 0..{max_marks}>, "grade_distribution": <array of {max_marks + 1} floats summing to 1.0>, "justification": "<one sentence>", "error_type": "<null|algebraic_error|missing_step|wrong_formula|presentation>"}}"""

        system_prompt = get_grading_system_prompt(subject, board, grade_level)
        system_prompt += REASONING_SYSTEM_PROMPT_SUFFIX

        result = call_llm(
            prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            call_type="reasoning_grading",
            api_key=api_key,
        )

        if not result["success"]:
            dist = [0.0] * (max_marks + 1)
            dist[0] = 1.0
            step_grades.append({
                "step_num": rs_num,
                "marks_awarded": 0,
                "max_marks": max_marks,
                "grade_distribution": dist,
                "justification": f"Reasoning evaluation failed: {result.get('error', 'Unknown')}",
                "error_type": None,
                "sympy_valid": sympy_result.get("valid") if sympy_result else None,
            })
            continue

        parsed = parse_json_response(result["response_text"])
        if not parsed or not isinstance(parsed, dict):
            dist = [0.0] * (max_marks + 1)
            dist[max_marks // 2] = 1.0
            step_grades.append({
                "step_num": rs_num,
                "marks_awarded": max_marks // 2,
                "max_marks": max_marks,
                "grade_distribution": dist,
                "justification": "Reasoning evaluation response could not be parsed.",
                "error_type": None,
                "sympy_valid": sympy_result.get("valid") if sympy_result else None,
            })
            continue

        awarded = min(max(parsed.get("marks_awarded", 0), 0), max_marks)
        grade_dist = parsed.get("grade_distribution", [])
        if len(grade_dist) != max_marks + 1:
            grade_dist = [0.0] * (max_marks + 1)
            grade_dist[awarded] = 1.0

        step_grades.append({
            "step_num": rs_num,
            "marks_awarded": awarded,
            "max_marks": max_marks,
            "grade_distribution": grade_dist,
            "justification": parsed.get("justification", ""),
            "error_type": parsed.get("error_type"),
            "sympy_valid": sympy_result.get("valid") if sympy_result else None,
        })

    # Aggregate
    total_marks = sum(sg["marks_awarded"] for sg in step_grades)
    max_marks_total = sum(sg["max_marks"] for sg in step_grades)

    # Confidence: SymPy validation boosts confidence
    sympy_validated = sum(1 for sg in step_grades if sg.get("sympy_valid") is True)
    error_count = sum(1 for sg in step_grades if sg.get("error_type"))
    total_steps = max(len(step_grades), 1)
    confidence = (1.0 - error_count / total_steps) + 0.1 * (sympy_validated / total_steps)
    confidence = min(confidence, 1.0)

    dist = [0.0] * (max_marks_total + 1) if max_marks_total > 0 else [1.0]
    if max_marks_total > 0:
        dist[min(total_marks, max_marks_total)] = 1.0

    justifications = [
        f"Step {sg['step_num']}: {sg['justification']}"
        for sg in step_grades if sg.get("justification")
    ]

    return {
        "type": "reasoning",
        "marks_awarded": total_marks,
        "max_marks": max_marks_total,
        "confidence": confidence,
        "grade_distribution": dist,
        "step_grades": step_grades,
        "justification": " | ".join(justifications),
    }
