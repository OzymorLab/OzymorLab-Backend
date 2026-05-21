"""
Diagram Client — Bridge between the Edexia-Backend and the DEIS Diagram-marker cluster.

This module submits diagram images to the DEIS Gateway for evaluation via its
YOLO + OCR + Graph Isomorphism pipeline, polls for results, and returns the
diagram score in a format that can be merged into the cumulative GradeResult.
"""
import time
import logging
import requests

from app.config import settings
from app.services.ingestion import generate_presigned_url

logger = logging.getLogger(__name__)


class DiagramEvaluationError(Exception):
    """Raised when the DEIS pipeline fails or times out."""
    pass


def submit_diagram_for_evaluation(
    file_key: str,
    rubric_relations: list[dict],
    question_id: str = "",
    max_marks: int = 5,
    submission_id: str = "",
    step_num: int = 0,
) -> str:
    """
    Submit a diagram image to the DEIS Gateway for AI evaluation.

    Args:
        file_key: S3 object key for the uploaded submission file.
        rubric_relations: List of expected label→region mappings from the rubric.
            Example: [{"label": "aorta", "region": "region_0"}, ...]
        question_id: The question/task ID for traceability.
        max_marks: Maximum marks for this diagram step.
        submission_id: The Edexia submission ID for traceability.
        step_num: The rubric step number this diagram corresponds to.

    Returns:
        deis_task_id: The DEIS task ID to poll for results.

    Raises:
        DiagramEvaluationError: If the DEIS Gateway is unreachable.
    """
    # Generate a presigned S3 URL so the DEIS workers can download the image
    image_url = generate_presigned_url(file_key)

    payload = {
        "image_url": image_url,
        "question_id": question_id,
        "submission_id": submission_id,
        "step_num": step_num,
        "rubric": {
            "max_marks": max_marks,
            "relations": rubric_relations,
        },
    }

    deis_url = f"{settings.DEIS_API_URL}/api/v1/diagram/evaluate"
    logger.info(f"Submitting diagram to DEIS: {deis_url} (file_key={file_key})")

    try:
        response = requests.post(deis_url, json=payload, timeout=10)
        response.raise_for_status()
    except requests.RequestException as e:
        logger.error(f"Failed to submit diagram to DEIS: {e}")
        raise DiagramEvaluationError(f"DEIS Gateway unreachable: {e}") from e

    result = response.json()
    deis_task_id = result.get("task_id")
    if not deis_task_id:
        raise DiagramEvaluationError("DEIS Gateway did not return a task_id")

    logger.info(f"Diagram submitted to DEIS, task_id={deis_task_id}")
    return deis_task_id


def poll_diagram_result(
    deis_task_id: str,
    timeout: int | None = None,
    interval: int | None = None,
) -> dict:
    """
    Poll the DEIS Gateway for the evaluation result until COMPLETED or timeout.

    Args:
        deis_task_id: The DEIS task ID returned by submit_diagram_for_evaluation.
        timeout: Max seconds to wait (defaults to settings.DEIS_POLL_TIMEOUT).
        interval: Seconds between polls (defaults to settings.DEIS_POLL_INTERVAL).

    Returns:
        DEIS evaluation result dict with keys:
            - diagram_detected (bool)
            - predicted_marks (int)
            - max_marks (int)
            - confidence (float)
            - missing_components (list[str])

    Raises:
        DiagramEvaluationError: If polling times out or DEIS returns an error.
    """
    timeout = timeout or settings.DEIS_POLL_TIMEOUT
    interval = interval or settings.DEIS_POLL_INTERVAL
    status_url = f"{settings.DEIS_API_URL}/api/v1/diagram/status/{deis_task_id}"

    deadline = time.time() + timeout
    logger.info(f"Polling DEIS for task {deis_task_id} (timeout={timeout}s)")

    while time.time() < deadline:
        try:
            response = requests.get(status_url, timeout=10)
            response.raise_for_status()
            result = response.json()
        except requests.RequestException as e:
            logger.warning(f"DEIS poll failed (retrying): {e}")
            time.sleep(interval)
            continue

        status = result.get("status", "UNKNOWN")

        if status == "COMPLETED":
            logger.info(
                f"DEIS task {deis_task_id} completed: "
                f"{result.get('predicted_marks')}/{result.get('max_marks')} "
                f"(confidence={result.get('confidence', 0):.2f})"
            )
            return result

        if status in ("FAILED", "ERROR"):
            raise DiagramEvaluationError(
                f"DEIS evaluation failed: {result.get('message', 'Unknown error')}"
            )

        # Still PROCESSING, wait and retry
        time.sleep(interval)

    raise DiagramEvaluationError(
        f"DEIS evaluation timed out after {timeout}s for task {deis_task_id}"
    )


def evaluate_diagram_step(
    file_key: str,
    rubric_step: dict,
    question_id: str = "",
    submission_id: str = "",
) -> dict:
    """
    High-level convenience function: submit a diagram step and return a
    grading result in the same format as grade_single_step().

    This is the main entry point called by the grading pipeline for
    diagram-type rubric steps.

    Args:
        file_key: S3 object key for the submission file.
        rubric_step: The rubric step dict (must have step_type="diagram").
        question_id: Task/question ID.
        submission_id: Submission ID for traceability.

    Returns:
        Step grade result dict compatible with the grading pipeline.
    """
    step_num = rubric_step.get("step_num", 0)
    max_marks = rubric_step.get("marks", 5)
    diagram_relations = rubric_step.get("diagram_relations", [])

    try:
        # Submit to DEIS
        deis_task_id = submit_diagram_for_evaluation(
            file_key=file_key,
            rubric_relations=diagram_relations,
            question_id=question_id,
            max_marks=max_marks,
            submission_id=submission_id,
            step_num=step_num,
        )

        # Poll for result
        deis_result = poll_diagram_result(deis_task_id)

        # Convert DEIS result to step grade format
        predicted_marks = min(max(deis_result.get("predicted_marks", 0), 0), max_marks)
        confidence = deis_result.get("confidence", 0.0)
        missing = deis_result.get("missing_components", [])

        # Build grade distribution
        dist = [0.0] * (max_marks + 1)
        dist[predicted_marks] = 1.0

        justification_parts = [f"Diagram evaluated by DEIS AI pipeline."]
        if missing:
            justification_parts.append(f"Missing components: {', '.join(missing)}")
        justification = " ".join(justification_parts)

        return {
            "step_num": step_num,
            "marks_awarded": predicted_marks,
            "max_marks": max_marks,
            "grade_distribution": dist,
            "justification": justification,
            "error_type": None,
            "sympy_valid": None,
            "sympy_error": None,
            "deis_task_id": deis_task_id,
            "deis_confidence": confidence,
            "_deis_raw": deis_result,  # Preserve raw DEIS payload for label pipeline
        }

    except DiagramEvaluationError as e:
        logger.error(f"Diagram evaluation failed for step {step_num}: {e}")
        # Graceful degradation: award 0 marks with an error justification
        dist = [0.0] * (max_marks + 1)
        dist[0] = 1.0
        return {
            "step_num": step_num,
            "marks_awarded": 0,
            "max_marks": max_marks,
            "grade_distribution": dist,
            "justification": f"Diagram evaluation failed: {e}",
            "error_type": "diagram_eval_failed",
            "sympy_valid": None,
            "sympy_error": None,
            "_deis_raw": None,  # No raw result available on failure
        }
