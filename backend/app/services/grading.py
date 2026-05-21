"""
Grading Orchestrator — Parallel Component-Based Multimodal Evaluation.

This is the central orchestrator that implements the proposed architecture:

  1. Question Decomposition → identify evaluation components
  2. Fan out to parallel pipelines:
     - Text Evaluation Pipeline       → theory marks
     - Diagram Evaluation Pipeline    → diagram marks (via DEIS)
     - Label Validation Pipeline      → label marks
     - Structured Reasoning Pipeline  → reasoning marks
  3. Score Fusion Engine              → cumulative grade
  4. Confidence Validation            → human review flagging

Each component is evaluated independently and then combined.
This mirrors real human evaluation behavior.
"""
import time
import logging

from app.services.question_decomposer import decompose_question
from app.services.text_evaluator import evaluate_text_component
from app.services.diagram_client import evaluate_diagram_step
from app.services.label_evaluator import evaluate_label_component
from app.services.reasoning_evaluator import evaluate_reasoning_component
from app.services.score_fusion import fuse_component_scores
from app.services.confidence_validator import validate_confidence

logger = logging.getLogger(__name__)


def grade_submission(
    rubric: dict,
    parsed_content: dict,
    temperature: float = 0.0,
    subject: str = "General",
    board: str = "Generic",
    grade_level: str = "Unknown",
    file_key: str | None = None,
    submission_id: str | None = None,
    user_gemini_key: str | None = None,
) -> dict:
    """
    Full multimodal grading pipeline for a single submission.

    Flow:
      1. Decompose question into evaluation components.
      2. Route each component to its dedicated pipeline.
      3. Fuse component scores into a cumulative grade.
      4. Validate confidence and flag for human review if needed.

    Args:
        rubric: The task rubric with steps, grading_notes, model.
        parsed_content: Parsed submission with steps, has_diagrams, etc.
        temperature: LLM grading temperature.
        subject: Subject name for system prompt.
        board: Board name (CBSE, ICSE, etc).
        grade_level: Grade level (Class 10, Class 12, etc).
        file_key: S3 object key for the submission file (needed for diagram eval).
        submission_id: Submission UUID for traceability.
        user_gemini_key: Optional BYOK Gemini API key for this user.

    Returns:
        Complete grade result dict with component breakdown and review status.
    """
    start_time = time.time()

    rubric_steps = rubric.get("steps", [])
    student_steps = parsed_content.get("steps", [])
    board_notes = rubric.get("grading_notes", "")
    question_text = rubric.get("description", "")

    # ═══════════════════════════════════════════════════════════
    # STEP 1: Question Decomposition
    # ═══════════════════════════════════════════════════════════
    logger.info("Step 1: Decomposing question into evaluation components...")
    components = decompose_question(rubric_steps, question_text)

    if not components:
        logger.warning("No components found. Falling back to single text component.")
        components = [{
            "type": "text",
            "description": "Full answer evaluation",
            "max_marks": sum(s.get("marks", 0) for s in rubric_steps),
            "rubric_steps": [s.get("step_num", i + 1) for i, s in enumerate(rubric_steps)],
            "source": "fallback",
        }]

    logger.info(
        f"Decomposed into {len(components)} components: "
        f"{[c['type'] for c in components]}"
    )

    # ═══════════════════════════════════════════════════════════
    # STEP 2: Parallel Pipeline Evaluation
    # ═══════════════════════════════════════════════════════════
    component_results = []
    deis_result_cache = None  # Cache DEIS result for both diagram and labels

    for component in components:
        ctype = component.get("type", "text")
        logger.info(
            f"Evaluating component: {ctype} "
            f"(max_marks={component.get('max_marks')}, "
            f"steps={component.get('rubric_steps')})"
        )

        if ctype == "text":
            # ── Text Evaluation Pipeline ──
            result = evaluate_text_component(
                component=component,
                rubric_steps=rubric_steps,
                student_steps=student_steps,
                board_notes=board_notes,
                temperature=temperature,
                subject=subject,
                board=board,
                grade_level=grade_level,
                api_key=user_gemini_key,
            )
            component_results.append(result)

        elif ctype == "diagram":
            # ── Diagram Evaluation Pipeline (via DEIS) ──
            if file_key:
                # Find the rubric step with diagram_relations
                component_step_nums = set(component.get("rubric_steps", []))
                diagram_rubric_step = next(
                    (s for s in rubric_steps if s.get("step_num") in component_step_nums),
                    None,
                )
                if diagram_rubric_step:
                    result = evaluate_diagram_step(
                        file_key=file_key,
                        rubric_step=diagram_rubric_step,
                        question_id=rubric.get("task_id", ""),
                        submission_id=submission_id or "",
                    )
                    # Cache the DEIS result for the label pipeline
                    deis_result_cache = result.get("_deis_raw", result)

                    component_results.append({
                        "type": "diagram",
                        "marks_awarded": result.get("marks_awarded", 0),
                        "max_marks": result.get("max_marks", component.get("max_marks", 0)),
                        "confidence": result.get("deis_confidence", 0.5),
                        "grade_distribution": result.get("grade_distribution", [1.0]),
                        "justification": result.get("justification", ""),
                    })
                else:
                    component_results.append(_zero_component("diagram", component))
            else:
                logger.warning("Diagram component found but no file_key provided.")
                component_results.append(_zero_component("diagram", component,
                    justification="Submission file not accessible for diagram evaluation."))

        elif ctype == "labels":
            # ── Label Validation Pipeline ──
            result = evaluate_label_component(
                component=component,
                deis_result=deis_result_cache,
                rubric_steps=rubric_steps,
            )
            component_results.append(result)

        elif ctype == "reasoning":
            # ── Structured Reasoning Pipeline ──
            result = evaluate_reasoning_component(
                component=component,
                rubric_steps=rubric_steps,
                student_steps=student_steps,
                board_notes=board_notes,
                temperature=temperature,
                subject=subject,
                board=board,
                grade_level=grade_level,
                api_key=user_gemini_key,
            )
            component_results.append(result)

        else:
            logger.warning(f"Unknown component type '{ctype}', treating as text.")
            result = evaluate_text_component(
                component=component,
                rubric_steps=rubric_steps,
                student_steps=student_steps,
                board_notes=board_notes,
                temperature=temperature,
                subject=subject,
                board=board,
                grade_level=grade_level,
                api_key=user_gemini_key,
            )
            component_results.append(result)

    # ═══════════════════════════════════════════════════════════
    # STEP 3: Score Fusion
    # ═══════════════════════════════════════════════════════════
    logger.info("Step 3: Fusing component scores...")
    fused_result = fuse_component_scores(component_results)

    # ═══════════════════════════════════════════════════════════
    # STEP 4: Confidence Validation
    # ═══════════════════════════════════════════════════════════
    logger.info("Step 4: Validating confidence...")
    fused_result = validate_confidence(fused_result)

    latency_ms = int((time.time() - start_time) * 1000)

    # Build final result
    return {
        "grade": fused_result["grade"],
        "max_grade": fused_result["max_grade"],
        "grade_distribution": fused_result["grade_distribution"],
        "confidence": fused_result["confidence"],
        "step_grades": _extract_step_grades(component_results),
        "component_grades": fused_result["component_grades"],
        "justification": fused_result["justification"],
        "review_status": fused_result.get("review_status", "AUTO_GRADED"),
        "review_reasons": fused_result.get("review_reasons", []),
        "flagged_components": fused_result.get("flagged_components", []),
        "question_decomposition": components,
        "llm_call_ids": [],
        "model_used": rubric.get("model", "gemini-2.5-pro"),
        "latency_ms": latency_ms,
    }


def _zero_component(ctype: str, component: dict, justification: str = "") -> dict:
    """Create a zero-score component result."""
    max_marks = component.get("max_marks", 0)
    dist = [0.0] * (max_marks + 1) if max_marks > 0 else [1.0]
    if dist:
        dist[0] = 1.0
    return {
        "type": ctype,
        "marks_awarded": 0,
        "max_marks": max_marks,
        "confidence": 0.0,
        "grade_distribution": dist,
        "justification": justification or f"No {ctype} evaluation performed.",
    }


def _extract_step_grades(component_results: list[dict]) -> list[dict]:
    """
    Extract individual step grades from component results for backward compatibility.
    The old schema expects a flat list of per-step grades.
    """
    step_grades = []
    for cr in component_results:
        if "step_grades" in cr and isinstance(cr["step_grades"], list):
            step_grades.extend(cr["step_grades"])
        else:
            # Create a synthetic step grade from the component
            step_grades.append({
                "step_num": 0,
                "marks_awarded": cr.get("marks_awarded", 0),
                "max_marks": cr.get("max_marks", 0),
                "grade_distribution": cr.get("grade_distribution", [1.0]),
                "justification": cr.get("justification", ""),
                "error_type": None,
                "sympy_valid": None,
                "sympy_error": None,
            })
    return step_grades
