"""
Text Evaluator — Dedicated Text Evaluation Pipeline.

Handles: explanations, definitions, theoretical reasoning, concept correctness.
Extracted from the original grading.py to serve as an independent evaluation component.

This pipeline:
  1. Aligns student text steps to the text-type rubric steps.
  2. Runs SymPy validation where applicable (derivation sub-type).
  3. Calls Gemini LLM for semantic grading of each step.
  4. Returns per-step results that feed into the Score Fusion Engine.
"""
import logging

from app.services.llm_client import (
    call_gemini, parse_json_response, build_step_grading_prompt,
    build_alignment_prompt, get_grading_system_prompt, ALIGNMENT_SYSTEM_PROMPT,
)
from app.services.sympy_validator import validate_expected_against_student

logger = logging.getLogger(__name__)


def align_steps(rubric_steps: list[dict], student_steps: list[dict]) -> list[dict]:
    """
    Align parsed student steps to rubric steps using LLM.
    Returns mapping of rubric_step → student_step(s).
    """
    if not rubric_steps or not student_steps:
        return [
            {
                "rubric_step": s.get("step_num", i + 1),
                "student_steps": [],
                "confidence": 0.0,
            }
            for i, s in enumerate(rubric_steps)
        ]

    # If there is only one rubric step, or it is a fallback catch-all step, map all student steps to it
    is_fallback = len(rubric_steps) == 1 or any(
        "auto-rubric generation failed" in s.get("description", "").lower()
        or "fallback" in s.get("description", "").lower()
        or "full answer" in s.get("description", "").lower()
        for s in rubric_steps
    )
    if is_fallback:
        return [
            {
                "rubric_step": s.get("step_num", i + 1),
                "student_steps": [st.get("step_num", j + 1) for j, st in enumerate(student_steps)],
                "confidence": 1.0,
            }
            for i, s in enumerate(rubric_steps)
        ]

    prompt = build_alignment_prompt(rubric_steps, student_steps)
    result = call_gemini(
        prompt,
        system_prompt=ALIGNMENT_SYSTEM_PROMPT,
        call_type="alignment",
        response_mime_type="application/json",
    )

    if not result["success"]:
        # Fallback: positional mapping
        return [
            {
                "rubric_step": i + 1,
                "student_steps": [i + 1] if i < len(student_steps) else [],
                "confidence": 0.5,
            }
            for i in range(len(rubric_steps))
        ]

    alignment = parse_json_response(result["response_text"])
    if not alignment or not isinstance(alignment, list):
        return [
            {
                "rubric_step": i + 1,
                "student_steps": [i + 1] if i < len(student_steps) else [],
                "confidence": 0.5,
            }
            for i in range(len(rubric_steps))
        ]

    return alignment


def grade_single_step(
    rubric_step: dict,
    student_step: dict | None,
    sympy_result: dict | None = None,
    board_notes: str = "",
    temperature: float = 0.0,
    subject: str = "General",
    board: str = "Generic",
    grade_level: str = "Unknown",
    api_key: str | None = None,
) -> dict:
    """
    Grade a single rubric step using the hybrid SymPy + LLM pipeline.
    Returns per-step grade result.
    """
    max_marks = rubric_step.get("marks", 1)

    # If no student step matched, award 0
    if student_step is None:
        dist = [0.0] * (max_marks + 1)
        dist[0] = 1.0
        return {
            "step_num": rubric_step.get("step_num", 0),
            "marks_awarded": 0,
            "max_marks": max_marks,
            "grade_distribution": dist,
            "justification": "No matching student answer found for this rubric step.",
            "error_type": "missing_step",
            "sympy_valid": None,
            "sympy_error": None,
        }

    prompt = build_step_grading_prompt(
        rubric_step, student_step, sympy_result, board_notes
    )
    system_prompt = get_grading_system_prompt(
        subject=subject, board=board, grade_level=grade_level
    )
    result = call_gemini(
        prompt,
        system_prompt=system_prompt,
        temperature=temperature,
        call_type="step_grading",
        api_key=api_key,
        response_mime_type="application/json",
    )

    if not result["success"]:
        dist = [0.0] * (max_marks + 1)
        dist[0] = 1.0
        return {
            "step_num": rubric_step.get("step_num", 0),
            "marks_awarded": 0,
            "max_marks": max_marks,
            "grade_distribution": dist,
            "justification": f"LLM grading failed: {result.get('error', 'Unknown')}",
            "error_type": None,
            "sympy_valid": sympy_result.get("valid") if sympy_result else None,
            "sympy_error": sympy_result.get("error") if sympy_result else None,
        }

    parsed = parse_json_response(result["response_text"])
    if not parsed or not isinstance(parsed, dict):
        dist = [0.0] * (max_marks + 1)
        dist[max_marks // 2] = 1.0  # uncertain → middle
        return {
            "step_num": rubric_step.get("step_num", 0),
            "marks_awarded": max_marks // 2,
            "max_marks": max_marks,
            "grade_distribution": dist,
            "justification": "LLM response could not be parsed.",
            "error_type": None,
            "sympy_valid": sympy_result.get("valid") if sympy_result else None,
            "sympy_error": sympy_result.get("error") if sympy_result else None,
        }

    # Normalize the grade distribution
    grade_dist = parsed.get("grade_distribution", [])
    if len(grade_dist) != max_marks + 1:
        grade_dist = [0.0] * (max_marks + 1)
        awarded = min(max(parsed.get("marks_awarded", 0), 0), max_marks)
        grade_dist[awarded] = 1.0
    else:
        total = sum(grade_dist)
        if total > 0:
            grade_dist = [x / total for x in grade_dist]
        else:
            grade_dist[0] = 1.0

    return {
        "step_num": rubric_step.get("step_num", 0),
        "marks_awarded": min(max(parsed.get("marks_awarded", 0), 0), max_marks),
        "max_marks": max_marks,
        "grade_distribution": grade_dist,
        "justification": parsed.get("justification", ""),
        "error_type": parsed.get("error_type"),
        "sympy_valid": sympy_result.get("valid") if sympy_result else None,
        "sympy_error": sympy_result.get("error") if sympy_result else None,
    }


def evaluate_text_component(
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
    Evaluate a text-type component. Main entry point for the Text Pipeline.

    Args:
        component: The decomposed component dict (type, rubric_steps, max_marks).
        rubric_steps: Full list of rubric steps (will be filtered to this component's steps).
        student_steps: Parsed student answer steps.
        board_notes: Board-specific grading notes.

    Returns:
        Component evaluation result dict.
    """
    # Filter rubric steps belonging to this component
    component_step_nums = set(component.get("rubric_steps", []))
    relevant_rubric_steps = [
        s for s in rubric_steps if s.get("step_num") in component_step_nums
    ]

    if not relevant_rubric_steps:
        return {
            "type": "text",
            "marks_awarded": 0,
            "max_marks": component.get("max_marks", 0),
            "confidence": 0.0,
            "grade_distribution": [1.0],
            "step_grades": [],
            "justification": "No rubric steps found for this text component.",
        }

    # Align student steps to this component's rubric steps
    alignment = align_steps(relevant_rubric_steps, student_steps)

    # Normalize alignment list to ensure all step identifiers are integers
    normalized_alignment = []
    for item in (alignment or []):
        r_val = item.get("rubric_step")
        if isinstance(r_val, str):
            try:
                r_val = int("".join(c for c in r_val if c.isdigit()))
            except ValueError:
                continue
        
        s_vals = []
        for s_val in item.get("student_steps", []):
            if isinstance(s_val, str):
                try:
                    s_vals.append(int("".join(c for c in s_val if c.isdigit())))
                except ValueError:
                    pass
            elif isinstance(s_val, (int, float)):
                s_vals.append(int(s_val))
                
        normalized_alignment.append({
            "rubric_step": r_val,
            "student_steps": s_vals,
            "confidence": item.get("confidence", 0.5)
        })
    alignment = normalized_alignment

    student_steps_by_num = {
        s.get("step_num", i + 1): s for i, s in enumerate(student_steps)
    }

    step_grades = []
    for rubric_step in relevant_rubric_steps:
        rs_num = rubric_step.get("step_num", 0)

        # Find aligned student steps
        aligned = next(
            (a for a in alignment if a.get("rubric_step") == rs_num), None
        )
        aligned_nums = aligned.get("student_steps", []) if aligned else []

        # Merge aligned student steps
        if aligned_nums:
            merged_text = "\n".join(
                student_steps_by_num[n].get("text", "")
                for n in aligned_nums
                if n in student_steps_by_num
            )
            merged_eqs = []
            for n in aligned_nums:
                if n in student_steps_by_num:
                    merged_eqs.extend(student_steps_by_num[n].get("equations", []))
            student_step = {
                "text": merged_text,
                "equations": merged_eqs,
                "step_num": aligned_nums[0],
            }
        else:
            student_step = None

        # SymPy validation for derivation steps
        sympy_result = None
        if (
            rubric_step.get("step_type") == "derivation"
            and rubric_step.get("expected_exprs")
            and student_step
            and student_step.get("equations")
        ):
            validations = validate_expected_against_student(
                rubric_step["expected_exprs"],
                student_step["equations"],
            )
            all_valid = all(v.get("valid") is True for v in validations)
            any_invalid = any(v.get("valid") is False for v in validations)
            if all_valid:
                sympy_result = {"valid": True, "error": None}
            elif any_invalid:
                errors = [v["error"] for v in validations if v.get("valid") is False]
                sympy_result = {"valid": False, "error": "; ".join(errors)}
            else:
                sympy_result = {"valid": None, "error": "Could not parse", "fallback": True}

        # LLM grading
        step_result = grade_single_step(
            rubric_step, student_step, sympy_result, board_notes, temperature,
            subject=subject, board=board, grade_level=grade_level, api_key=api_key,
        )
        step_grades.append(step_result)

    # Aggregate
    total_marks = sum(sg["marks_awarded"] for sg in step_grades)
    max_marks = sum(sg["max_marks"] for sg in step_grades)
    confidence = _compute_text_confidence(step_grades)

    # Build combined distribution
    combined_dist = [0.0] * (max_marks + 1)
    if max_marks > 0:
        combined_dist[min(total_marks, max_marks)] = 1.0

    justifications = [
        f"Step {sg['step_num']}: {sg['justification']}"
        for sg in step_grades if sg.get("justification")
    ]

    return {
        "type": "text",
        "marks_awarded": total_marks,
        "max_marks": max_marks,
        "confidence": confidence,
        "grade_distribution": combined_dist,
        "step_grades": step_grades,
        "justification": " | ".join(justifications),
    }


def _compute_text_confidence(step_grades: list[dict]) -> float:
    """Compute text evaluation confidence based on step grade quality."""
    if not step_grades:
        return 0.0

    # Steps with errors or missing content reduce confidence
    error_count = sum(1 for sg in step_grades if sg.get("error_type"))
    total = len(step_grades)

    base_confidence = 1.0 - (error_count / max(total, 1))

    # SymPy validated steps boost confidence
    sympy_validated = sum(
        1 for sg in step_grades if sg.get("sympy_valid") is True
    )
    sympy_boost = 0.1 * (sympy_validated / max(total, 1))

    return min(base_confidence + sympy_boost, 1.0)
