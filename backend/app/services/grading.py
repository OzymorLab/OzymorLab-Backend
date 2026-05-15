"""
Grading service — hybrid SymPy + LLM grading pipeline orchestrator.
Implements the 6-step grading flow from the technical proposal.
"""
import logging
import numpy as np
from scipy.signal import fftconvolve

from app.services.llm_client import (
    call_gemini, parse_json_response, build_step_grading_prompt,
    build_alignment_prompt, GRADING_SYSTEM_PROMPT, ALIGNMENT_SYSTEM_PROMPT,
)
from app.services.sympy_validator import validate_expected_against_student

logger = logging.getLogger(__name__)


def align_steps(rubric_steps: list[dict], student_steps: list[dict]) -> list[dict]:
    """
    STEP 2: Align parsed student steps to rubric steps using LLM.
    Returns mapping of rubric_step → student_step(s).
    """
    if not rubric_steps or not student_steps:
        return [{"rubric_step": s.get("step_num", i+1), "student_steps": [], "confidence": 0.0}
                for i, s in enumerate(rubric_steps)]

    prompt = build_alignment_prompt(rubric_steps, student_steps)
    result = call_gemini(prompt, system_prompt=ALIGNMENT_SYSTEM_PROMPT, call_type="alignment")

    if not result["success"]:
        # Fallback: positional mapping
        return [{"rubric_step": i+1, "student_steps": [i+1] if i < len(student_steps) else [], "confidence": 0.5}
                for i in range(len(rubric_steps))]

    alignment = parse_json_response(result["response_text"])
    if not alignment or not isinstance(alignment, list):
        return [{"rubric_step": i+1, "student_steps": [i+1] if i < len(student_steps) else [], "confidence": 0.5}
                for i in range(len(rubric_steps))]

    return alignment


def grade_single_step(
    rubric_step: dict,
    student_step: dict | None,
    sympy_result: dict | None = None,
    board_notes: str = "",
    temperature: float = 0.0,
) -> dict:
    """
    STEP 3: Grade a single rubric step using the hybrid pipeline.
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

    prompt = build_step_grading_prompt(rubric_step, student_step, sympy_result, board_notes)
    result = call_gemini(prompt, system_prompt=GRADING_SYSTEM_PROMPT,
                         temperature=temperature, call_type="step_grading")

    if not result["success"]:
        dist = [0.0] * (max_marks + 1)
        dist[0] = 1.0
        return {
            "step_num": rubric_step.get("step_num", 0),
            "marks_awarded": 0, "max_marks": max_marks,
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
            "marks_awarded": max_marks // 2, "max_marks": max_marks,
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
        "llm_call_id": None,  # filled by caller
    }


def convolve_distributions(step_distributions: list[list[float]]) -> list[float]:
    """
    STEP 4: Compute the submission-level grade distribution by convolving per-step distributions.
    This is mathematically precise — represents true joint uncertainty.
    """
    if not step_distributions:
        return [1.0]

    result = np.array(step_distributions[0], dtype=float)
    for dist in step_distributions[1:]:
        result = fftconvolve(result, np.array(dist, dtype=float), mode="full")

    # Normalize
    total = result.sum()
    if total > 0:
        result = result / total

    return result.tolist()


def compute_confidence(grade_distribution: list[float]) -> float:
    """Compute confidence as 1 - normalized entropy of the distribution."""
    dist = np.array(grade_distribution, dtype=float)
    dist = dist[dist > 0]  # remove zeros for entropy calc
    if len(dist) <= 1:
        return 1.0
    entropy = -np.sum(dist * np.log2(dist))
    max_entropy = np.log2(len(grade_distribution))
    if max_entropy == 0:
        return 1.0
    return float(1.0 - (entropy / max_entropy))


def grade_submission(
    rubric: dict,
    parsed_content: dict,
    temperature: float = 0.0,
) -> dict:
    """
    Full grading pipeline for a single submission.

    Steps:
    1. Load rubric (passed in)
    2. Align student steps to rubric steps
    3. For each rubric step: SymPy validation → LLM grading
    4. Convolve step distributions → total distribution
    5. Return complete grade result

    Args:
        rubric: The task rubric with steps
        parsed_content: The parsed submission content
        temperature: Grading temperature

    Returns:
        Complete grade result dict
    """
    import time
    start_time = time.time()

    rubric_steps = rubric.get("steps", [])
    student_steps = parsed_content.get("steps", [])
    board_notes = rubric.get("grading_notes", "")

    # STEP 2: Align steps
    alignment = align_steps(rubric_steps, student_steps)

    # STEP 3: Grade each rubric step
    step_grades = []
    llm_call_ids = []
    student_steps_by_num = {s.get("step_num", i+1): s for i, s in enumerate(student_steps)}

    for rubric_step in rubric_steps:
        rs_num = rubric_step.get("step_num", 0)

        # Find aligned student steps
        aligned = next((a for a in alignment if a.get("rubric_step") == rs_num), None)
        aligned_nums = aligned.get("student_steps", []) if aligned else []

        # Merge aligned student steps into one
        if aligned_nums:
            merged_text = "\n".join(
                student_steps_by_num[n].get("text", "") for n in aligned_nums if n in student_steps_by_num
            )
            merged_eqs = []
            for n in aligned_nums:
                if n in student_steps_by_num:
                    merged_eqs.extend(student_steps_by_num[n].get("equations", []))
            student_step = {"text": merged_text, "equations": merged_eqs, "step_num": aligned_nums[0]}
        else:
            student_step = None

        # STEP 3a: SymPy validation for derivation steps
        sympy_result = None
        if (rubric_step.get("step_type") == "derivation"
                and rubric_step.get("expected_exprs")
                and student_step and student_step.get("equations")):
            validations = validate_expected_against_student(
                rubric_step["expected_exprs"],
                student_step["equations"],
            )
            # Use the overall validation result
            all_valid = all(v.get("valid") is True for v in validations)
            any_invalid = any(v.get("valid") is False for v in validations)
            if all_valid:
                sympy_result = {"valid": True, "error": None}
            elif any_invalid:
                errors = [v["error"] for v in validations if v.get("valid") is False]
                sympy_result = {"valid": False, "error": "; ".join(errors)}
            else:
                sympy_result = {"valid": None, "error": "Could not parse expressions", "fallback": True}

        # STEP 3b + 3c: LLM grading
        step_result = grade_single_step(
            rubric_step, student_step, sympy_result, board_notes, temperature
        )
        step_grades.append(step_result)

    # STEP 4: Aggregate distributions via convolution
    step_distributions = [sg["grade_distribution"] for sg in step_grades]
    total_distribution = convolve_distributions(step_distributions)

    # Calculate totals
    total_grade = sum(sg["marks_awarded"] for sg in step_grades)
    max_grade = sum(sg["max_marks"] for sg in step_grades)
    confidence = compute_confidence(total_distribution)

    # Overall justification
    justifications = [f"Step {sg['step_num']}: {sg['justification']}" for sg in step_grades if sg.get("justification")]
    overall_justification = " | ".join(justifications)

    latency_ms = int((time.time() - start_time) * 1000)

    return {
        "grade": total_grade,
        "max_grade": max_grade,
        "grade_distribution": total_distribution,
        "confidence": confidence,
        "step_grades": step_grades,
        "justification": overall_justification,
        "llm_call_ids": llm_call_ids,
        "model_used": rubric.get("model", "gemini-2.5-pro"),
        "latency_ms": latency_ms,
    }
