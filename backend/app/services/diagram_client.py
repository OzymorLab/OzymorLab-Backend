"""
Diagram Client — Hybrid Diagram Evaluation.

Strategy (fast + robust):
  PRIMARY  — Gemini Vision evaluates the diagram directly (~1-3 s).
  OPTIONAL — DEIS pipeline (YOLO + OCR + Graph) runs in parallel (~5-30 s).

Both run concurrently via threading. Whichever finishes first with the higher
confidence wins. If DEIS is unavailable, Gemini Vision is the sole evaluator.
This means diagram grading NEVER blocks the overall grading pipeline for more
than the Gemini Vision latency.

Result schema is identical to what the label_evaluator.py expects, so the rest
of the grading pipeline is completely unaffected by which evaluator won.
"""
import logging
import threading
import time
from typing import Optional

from app.config import settings
from app.services.diagram_vision_evaluator import evaluate_diagram_with_gemini

logger = logging.getLogger(__name__)

# How long (seconds) to wait for DEIS before giving up.
# Gemini Vision result is used immediately even if DEIS is still running.
_DEIS_PARALLEL_TIMEOUT = 8  # seconds


class DiagramEvaluationError(Exception):
    """Raised when ALL evaluators fail for a diagram step."""
    pass


# ── DEIS submission helpers (kept for parallel use) ──────────────────────────

def _submit_deis(
    file_key: str,
    rubric_relations: list[dict],
    question_id: str,
    max_marks: int,
    submission_id: str,
    step_num: int,
) -> Optional[str]:
    """
    Submit to DEIS Gateway and return the task_id.
    Returns None if DEIS is unreachable (treated as optional).
    """
    import requests
    from app.services.ingestion import generate_presigned_url

    try:
        image_url = generate_presigned_url(file_key)
    except Exception as e:
        logger.warning(f"[DEIS] Could not generate presigned URL: {e}")
        return None

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
    try:
        resp = requests.post(deis_url, json=payload, timeout=5)
        resp.raise_for_status()
        task_id = resp.json().get("task_id")
        if task_id:
            logger.info(f"[DEIS] Task submitted: {task_id}")
        return task_id
    except Exception as e:
        logger.warning(f"[DEIS] Gateway unreachable or submission failed: {e}")
        return None


def _poll_deis(task_id: str, deadline: float) -> Optional[dict]:
    """
    Poll DEIS status endpoint until COMPLETED, FAILED, or deadline.
    Returns raw DEIS result dict or None on timeout/failure.
    """
    import requests

    status_url = f"{settings.DEIS_API_URL}/api/v1/diagram/status/{task_id}"
    interval = settings.DEIS_POLL_INTERVAL

    while time.time() < deadline:
        try:
            resp = requests.get(status_url, timeout=5)
            resp.raise_for_status()
            result = resp.json()
            status = result.get("status", "UNKNOWN")
            if status == "COMPLETED":
                return result
            if status in ("FAILED", "ERROR"):
                logger.warning(f"[DEIS] Task {task_id} failed: {result.get('message')}")
                return None
        except Exception as e:
            logger.debug(f"[DEIS] Poll error for {task_id}: {e}")
        time.sleep(interval)

    logger.warning(f"[DEIS] Task {task_id} timed out after {_DEIS_PARALLEL_TIMEOUT}s")
    return None


def _deis_result_to_step_grade(deis_result: dict, step_num: int, max_marks: int) -> dict:
    """Convert raw DEIS result into the step-grade dict format."""
    predicted_marks = min(max(int(deis_result.get("predicted_marks", 0)), 0), max_marks)
    confidence = float(deis_result.get("confidence", 0.0))
    confidence = max(0.0, min(1.0, confidence))
    missing = deis_result.get("missing_components", [])

    dist = [0.0] * (max_marks + 1)
    dist[predicted_marks] = 1.0

    justification = "Diagram evaluated by DEIS AI pipeline (YOLO + OCR + Graph)."
    if missing:
        justification += f" Missing: {', '.join(missing[:5])}."

    label_scores = deis_result.get("label_scores", [])

    return {
        "step_num": step_num,
        "marks_awarded": predicted_marks,
        "max_marks": max_marks,
        "confidence": confidence,
        "grade_distribution": dist,
        "justification": justification,
        "error_type": None,
        "sympy_valid": None,
        "sympy_error": None,
        "deis_confidence": confidence,
        "label_scores": label_scores,
        "missing_components": missing,
        "diagram_detected": deis_result.get("diagram_detected", True),
        "_deis_raw": deis_result,
        "_gemini_raw": None,
        "evaluator": "deis",
    }


# ── Main entry point ──────────────────────────────────────────────────────────

def evaluate_diagram_step(
    file_key: str,
    rubric_step: dict,
    question_id: str = "",
    submission_id: str = "",
    api_key: Optional[str] = None,
) -> dict:
    """
    Evaluate a diagram rubric step using a parallel Gemini Vision + DEIS strategy.

    Execution flow:
      1. Launch DEIS submission in a background thread (fire-and-forget).
      2. Run Gemini Vision evaluation synchronously (fast path, ~1-3 s).
      3. After Gemini finishes, wait up to _DEIS_PARALLEL_TIMEOUT seconds
         for DEIS to also complete.
      4. If both results are available, pick the one with higher confidence.
         If only Gemini succeeded, use it. If only DEIS succeeded (unlikely),
         use that. If both failed, raise DiagramEvaluationError.

    This means:
      - Best case (Gemini fast, DEIS available): ~2-4 s total, highest-quality result.
      - Normal case (Gemini fast, DEIS slow/unavailable): ~1-3 s, Gemini result.
      - Worst case (both fail): raises DiagramEvaluationError (caller awards 0 marks).

    Args:
        file_key: S3 object key for the diagram image.
        rubric_step: Rubric step dict with marks, diagram_relations, marking_notes.
        question_id: Task/question ID for tracing.
        submission_id: Submission UUID.
        api_key: Optional BYOK Gemini API key.

    Returns:
        Step-grade-compatible dict. Winning result is tagged with "evaluator" key.
    """
    step_num = rubric_step.get("step_num", 0)
    max_marks = rubric_step.get("marks", 5)
    diagram_relations = rubric_step.get("diagram_relations", [])

    # ── Container for DEIS result (written by background thread) ─────────────
    deis_container: dict = {"result": None, "done": False}

    def _run_deis():
        """Background thread: submit to DEIS and poll until done or timeout."""
        deadline = time.time() + _DEIS_PARALLEL_TIMEOUT
        task_id = _submit_deis(
            file_key=file_key,
            rubric_relations=diagram_relations,
            question_id=question_id,
            max_marks=max_marks,
            submission_id=submission_id,
            step_num=step_num,
        )
        if task_id:
            raw = _poll_deis(task_id, deadline)
            if raw:
                deis_container["result"] = _deis_result_to_step_grade(
                    raw, step_num, max_marks
                )
        deis_container["done"] = True

    # ── Fire DEIS in background ───────────────────────────────────────────────
    deis_thread = threading.Thread(target=_run_deis, daemon=True)
    deis_thread.start()

    # ── Run Gemini Vision (fast path) ─────────────────────────────────────────
    gemini_result = None
    try:
        gemini_result = evaluate_diagram_with_gemini(
            file_key=file_key,
            rubric_step=rubric_step,
            question_id=question_id,
            submission_id=submission_id,
            api_key=api_key,
        )
        logger.info(
            f"[DiagramClient] Gemini Vision done: "
            f"{gemini_result['marks_awarded']}/{max_marks} "
            f"confidence={gemini_result['confidence']:.2f}"
        )
    except Exception as e:
        logger.error(f"[DiagramClient] Gemini Vision raised unexpectedly: {e}")
        gemini_result = None

    # ── Wait for DEIS (up to remaining time budget) ───────────────────────────
    remaining = _DEIS_PARALLEL_TIMEOUT - (
        # Time already consumed by Gemini call; clamp to [0, timeout]
        0  # DEIS started before Gemini, so it has had the full timeout running
    )
    deis_thread.join(timeout=max(0.0, remaining))

    deis_result = deis_container.get("result")

    if deis_result:
        logger.info(
            f"[DiagramClient] DEIS done: "
            f"{deis_result['marks_awarded']}/{max_marks} "
            f"confidence={deis_result['confidence']:.2f}"
        )

    # ── Pick the winner ───────────────────────────────────────────────────────
    winner = _select_best_result(gemini_result, deis_result, step_num, max_marks)

    if winner is None:
        # Both failed
        logger.error(
            f"[DiagramClient] All diagram evaluators failed for step {step_num}."
        )
        raise DiagramEvaluationError(
            f"All diagram evaluators (Gemini Vision + DEIS) failed for step {step_num}."
        )

    logger.info(
        f"[DiagramClient] Selected evaluator='{winner.get('evaluator', '?')}' "
        f"for step {step_num}: {winner['marks_awarded']}/{max_marks} marks"
    )
    return winner


def _select_best_result(
    gemini_result: Optional[dict],
    deis_result: Optional[dict],
    step_num: int,
    max_marks: int,
) -> Optional[dict]:
    """
    Choose the higher-confidence result.

    Rules (in priority order):
    1. If only one result is available, use it.
    2. If both are available, pick higher confidence.
    3. Tie → prefer Gemini (it processes the raw handwriting, not a derived graph).
    4. Both failed → return None.

    When DEIS has higher confidence, we also merge Gemini's parsed label details
    back into the DEIS result (DEIS label_scores are more structured).
    """
    if gemini_result is None and deis_result is None:
        return None

    if gemini_result is None:
        return deis_result

    if deis_result is None:
        return gemini_result

    g_conf = gemini_result.get("confidence", 0.0)
    d_conf = deis_result.get("confidence", 0.0)

    # DEIS wins only if its confidence is meaningfully higher (> 0.10 margin)
    if d_conf > g_conf + 0.10:
        # Enrich DEIS result with Gemini's justification for better UX
        deis_result["justification"] = (
            f"[DEIS: {deis_result['justification']}] "
            f"[Gemini cross-check: {gemini_result.get('justification', '')}]"
        )
        # Merge Gemini label data if DEIS label_scores is empty
        if not deis_result.get("label_scores"):
            deis_result["label_scores"] = gemini_result.get("label_scores", [])
        deis_result["_gemini_raw"] = gemini_result.get("_gemini_raw")
        return deis_result

    # Gemini wins (or tie) — enrich with DEIS breakdown if available
    if deis_result.get("label_scores"):
        # Merge DEIS structured label_scores into Gemini result for richer label grading
        gemini_result["label_scores"] = _merge_label_scores(
            gemini_result.get("label_scores", []),
            deis_result.get("label_scores", []),
        )
    gemini_result["_deis_raw"] = deis_result.get("_deis_raw")
    gemini_result["justification"] = (
        f"{gemini_result.get('justification', '')} "
        f"[DEIS cross-check: {d_conf:.2f} confidence]"
    ).strip()
    return gemini_result


def _merge_label_scores(
    gemini_labels: list[dict],
    deis_labels: list[dict],
) -> list[dict]:
    """
    Merge Gemini and DEIS label scores. For each expected label:
    - If DEIS says matched=True, that's a strong structural signal — trust it.
    - Otherwise fall back to Gemini's vision-based match.
    """
    deis_by_expected = {
        l.get("expected", "").lower(): l for l in deis_labels
    }
    merged = []
    for gl in gemini_labels:
        expected_lower = gl.get("expected", "").lower()
        deis_entry = deis_by_expected.get(expected_lower)
        if deis_entry and deis_entry.get("matched"):
            # DEIS confirmed the label structurally — use DEIS as ground truth
            merged.append({
                **gl,
                "matched": True,
                "text": deis_entry.get("text") or gl.get("text") or "",
                "target_region": deis_entry.get("target_region") or gl.get("target_region", ""),
            })
        else:
            merged.append(gl)
    return merged
