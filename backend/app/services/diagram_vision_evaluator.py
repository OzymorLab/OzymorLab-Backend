"""
Diagram Vision Evaluator — Gemini Vision-based diagram grading.

This is the PRIMARY diagram evaluator. It uses Gemini 2.5 Pro Vision to:
  1. Detect all labels visible in the diagram.
  2. Assess structural correctness (arrows pointing to correct regions).
  3. Match detected labels against the rubric's expected labels.
  4. Award marks with full justification.

It is fast (~1-3s per diagram), works on messy/freehand drawings, and requires
zero infrastructure (no YOLO, no Kafka, no Redis).

The DEIS pipeline (YOLOv8 + graph isomorphism) runs optionally in parallel
and its result is used only when its confidence exceeds Gemini's result.

Output schema is fully compatible with label_evaluator.py — the `label_scores`
key matches exactly what the DEIS scoring worker returns.
"""
import base64
import json
import logging
import time
from typing import Optional

from app.services.llm_client import get_client, parse_json_response, call_llm
from app.services.ingestion import download_file, generate_presigned_url
from app.config import settings

logger = logging.getLogger(__name__)

# ── Gemini diagram grading prompt ──────────────────────────────────────────────
_DIAGRAM_SYSTEM_PROMPT = """You are an expert examiner grading handwritten student diagrams.
You receive an image of a student's handwritten diagram and a rubric that describes what the 
correct diagram should contain — which labels should appear and where they should point.

You output ONLY valid JSON — no preamble, no markdown, no code fences.

Your job:
1. Identify every label visible in the student's diagram (read handwriting carefully).
2. For each rubric-required label, determine if the student has drawn it correctly:
   - Label must be present (legible, even if slightly misspelled).
   - Label must point to / annotate the correct anatomical region or component.
3. Assess overall structural correctness of the diagram.
4. Assign marks proportionally.

Be lenient with spelling (OCR artifacts and handwriting quirks). 
"aorta" and "Aorta" and "aorts" are all the same. 
Focus on whether the student has the correct BIOLOGICAL/SCIENTIFIC understanding."""

_DIAGRAM_PROMPT_TEMPLATE = """RUBRIC FOR THIS DIAGRAM STEP:
Maximum marks: {max_marks}
Required labels and their correct positions:
{relations_text}

Marking guidance: {marking_notes}

TASK:
Examine the student's handwritten diagram carefully.
For each required label above, determine if the student has correctly placed it.

Return JSON in EXACTLY this format:
{{
  "diagram_detected": true,
  "overall_structure_correct": <true|false>,
  "predicted_marks": <integer 0..{max_marks}>,
  "confidence": <float 0.0..1.0>,
  "label_results": [
    {{
      "expected": "<rubric label name>",
      "detected_text": "<what student actually wrote, or null if absent>",
      "matched": <true|false>,
      "correctly_placed": <true|false>,
      "target_region": "<region description>",
      "notes": "<brief note about this label>"
    }}
  ],
  "missing_components": ["<list of missing or incorrectly drawn components>"],
  "overall_justification": "<one or two sentence overall assessment>"
}}

Rules for predicted_marks:
- Full marks only if all required labels are correct AND structure is sound.
- Proportional partial marks for partially correct diagrams.
- Never award more than {max_marks}.
- If diagram is absent or unrecognisable, award 0."""


def _build_relations_text(diagram_relations: list[dict]) -> str:
    """Format rubric diagram_relations into human-readable text for the prompt."""
    if not diagram_relations:
        return "  (No specific label requirements — assess overall diagram quality)"
    lines = []
    for i, rel in enumerate(diagram_relations, 1):
        label = rel.get("label", "?")
        region = rel.get("region", "?")
        notes = rel.get("notes", "")
        line = f"  {i}. Label '{label}' must point to / annotate '{region}'"
        if notes:
            line += f" ({notes})"
        lines.append(line)
    return "\n".join(lines)


def evaluate_diagram_with_gemini(
    file_key: str,
    rubric_step: dict,
    question_id: str = "",
    submission_id: str = "",
    api_key: Optional[str] = None,
) -> dict:
    """
    Evaluate a diagram step using Gemini Vision.

    Args:
        file_key: S3 object key (or diagram crop key from parse step).
        rubric_step: Rubric step dict with marks, diagram_relations, marking_notes.
        question_id: Task/question ID for traceability.
        submission_id: Submission ID for traceability.
        api_key: Optional BYOK Gemini API key.

    Returns:
        Step-grade-compatible dict with label_scores for label_evaluator.py.
        Keys: marks_awarded, max_marks, confidence, grade_distribution,
              justification, label_scores, missing_components, _gemini_raw.
    """
    step_num = rubric_step.get("step_num", 0)
    max_marks = rubric_step.get("marks", 5)
    diagram_relations = rubric_step.get("diagram_relations", [])
    marking_notes = rubric_step.get("marking_notes", "Award marks for each correctly identified and placed label.")

    start_time = time.time()
    logger.info(
        f"[DiagramVision] Evaluating step {step_num} via Gemini Vision "
        f"(max_marks={max_marks}, labels={len(diagram_relations)}, "
        f"submission={submission_id})"
    )

    # ── Download the image ───────────────────────────────────────────────────
    try:
        image_bytes = download_file(file_key)
    except Exception as e:
        logger.error(f"[DiagramVision] Failed to download file {file_key}: {e}")
        return _error_result(step_num, max_marks, f"Image download failed: {e}")

    # ── Detect MIME type ─────────────────────────────────────────────────────
    mime_type = _detect_mime(file_key, image_bytes)

    # ── Build prompt ─────────────────────────────────────────────────────────
    relations_text = _build_relations_text(diagram_relations)
    prompt_text = _DIAGRAM_PROMPT_TEMPLATE.format(
        max_marks=max_marks,
        relations_text=relations_text,
        marking_notes=marking_notes,
    )

    # ── Call LLM Vision ──────────────────────────────────────────────────────
    try:
        result = call_llm(
            prompt=prompt_text,
            system_prompt=_DIAGRAM_SYSTEM_PROMPT,
            temperature=0.0,
            max_tokens=1024,
            image_bytes=image_bytes,
            image_mime=mime_type,
            call_type="diagram_vision_eval",
            api_key=api_key,
        )
        if not result["success"]:
            logger.error(f"[DiagramVision] LLM call failed: {result.get('error')}")
            return _error_result(step_num, max_marks, f"LLM call failed: {result.get('error')}")
        raw_text = result["response_text"]
    except Exception as e:
        logger.error(f"[DiagramVision] LLM call failed: {e}")
        return _error_result(step_num, max_marks, f"LLM call failed: {e}")

    latency_ms = int((time.time() - start_time) * 1000)

    # ── Parse JSON response ───────────────────────────────────────────────────
    parsed = parse_json_response(raw_text)
    if not parsed or not isinstance(parsed, dict):
        logger.warning(
            f"[DiagramVision] Could not parse Gemini response for step {step_num}. "
            f"Raw: {raw_text[:300]}"
        )
        return _error_result(step_num, max_marks, "Gemini Vision response could not be parsed.")

    # ── Extract and clamp marks ───────────────────────────────────────────────
    predicted_marks = min(max(int(parsed.get("predicted_marks", 0)), 0), max_marks)
    confidence = float(parsed.get("confidence", 0.5))
    confidence = max(0.0, min(1.0, confidence))

    # ── Build label_scores (compatible with label_evaluator.py) ──────────────
    label_scores = []
    for lr in parsed.get("label_results", []):
        label_scores.append({
            "expected": lr.get("expected", ""),
            "text": lr.get("detected_text") or "",
            "matched": bool(lr.get("matched", False)),
            "target_region": lr.get("target_region", ""),
        })

    # If no label_results in response but we have relations, build from relations
    if not label_scores and diagram_relations:
        for rel in diagram_relations:
            label_scores.append({
                "expected": rel.get("label", ""),
                "text": "",
                "matched": False,
                "target_region": rel.get("region", ""),
            })

    # ── Grade distribution ────────────────────────────────────────────────────
    dist = [0.0] * (max_marks + 1)
    dist[predicted_marks] = 1.0

    # ── Justification ─────────────────────────────────────────────────────────
    missing = parsed.get("missing_components", [])
    overall_just = parsed.get("overall_justification", "Diagram evaluated by Gemini Vision.")
    if missing:
        overall_just += f" Missing: {', '.join(missing[:5])}."

    logger.info(
        f"[DiagramVision] Step {step_num}: {predicted_marks}/{max_marks} marks, "
        f"confidence={confidence:.2f}, latency={latency_ms}ms"
    )

    return {
        "step_num": step_num,
        "marks_awarded": predicted_marks,
        "max_marks": max_marks,
        "confidence": confidence,
        "grade_distribution": dist,
        "justification": overall_just,
        "error_type": None,
        "sympy_valid": None,
        "sympy_error": None,
        "deis_confidence": confidence,
        # label_scores is consumed by label_evaluator.py
        "label_scores": label_scores,
        "missing_components": missing,
        "diagram_detected": parsed.get("diagram_detected", True),
        "_gemini_raw": parsed,
        # Make it compatible with the _deis_raw key that grading.py caches
        "_deis_raw": {
            "label_scores": label_scores,
            "missing_components": missing,
            "confidence": confidence,
            "predicted_marks": predicted_marks,
            "max_marks": max_marks,
        },
        "latency_ms": latency_ms,
        "evaluator": "gemini_vision",
    }


def _detect_mime(file_key: str, image_bytes: bytes) -> str:
    """Infer MIME type from file key extension, fallback to magic bytes."""
    key_lower = file_key.lower()
    if key_lower.endswith(".png"):
        return "image/png"
    if key_lower.endswith((".jpg", ".jpeg")):
        return "image/jpeg"
    if key_lower.endswith(".webp"):
        return "image/webp"
    if key_lower.endswith(".gif"):
        return "image/gif"
    # Magic bytes fallback
    if image_bytes[:4] == b"\x89PNG":
        return "image/png"
    if image_bytes[:2] in (b"\xff\xd8",):
        return "image/jpeg"
    # Default — Gemini handles most image formats
    return "image/png"


def _error_result(step_num: int, max_marks: int, reason: str) -> dict:
    """Return a zero-score result with an error justification."""
    dist = [0.0] * (max_marks + 1)
    dist[0] = 1.0
    return {
        "step_num": step_num,
        "marks_awarded": 0,
        "max_marks": max_marks,
        "confidence": 0.0,
        "grade_distribution": dist,
        "justification": f"Diagram evaluation failed: {reason}",
        "error_type": "diagram_eval_failed",
        "sympy_valid": None,
        "sympy_error": None,
        "deis_confidence": 0.0,
        "label_scores": [],
        "missing_components": [],
        "diagram_detected": False,
        "_gemini_raw": None,
        "_deis_raw": None,
        "latency_ms": 0,
        "evaluator": "gemini_vision_error",
    }
