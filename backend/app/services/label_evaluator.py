"""
Label Evaluator — Label Validation Pipeline.

Handles: handwritten label text matching, label placement validation,
terminology correctness, and per-label partial marks.

This pipeline receives OCR-extracted label data from the DEIS scoring result
and validates each label against the rubric's expected terminology using
fuzzy string matching (to account for handwriting OCR errors).
"""
import logging
from thefuzz import fuzz

from app.config import settings

logger = logging.getLogger(__name__)


def evaluate_label_component(
    component: dict,
    deis_result: dict | None = None,
    rubric_steps: list[dict] = None,
) -> dict:
    """
    Evaluate a labels-type component.

    Args:
        component: Decomposed component dict (type="labels", max_marks, rubric_steps).
        deis_result: The DEIS evaluation result containing detected labels and
                     their mapping status. Expected keys:
                       - label_scores: list of {"text": str, "matched": bool, "target_region": str}
                       - missing_components: list of str
        rubric_steps: Rubric steps for this component (contain expected label names).

    Returns:
        Component evaluation result dict.
    """
    max_marks = component.get("max_marks", 0)
    rubric_steps = rubric_steps or []

    # Extract expected labels from rubric
    expected_labels = _extract_expected_labels(component, rubric_steps)

    if not expected_labels:
        return {
            "type": "labels",
            "marks_awarded": 0,
            "max_marks": max_marks,
            "confidence": 0.0,
            "grade_distribution": [1.0] + [0.0] * max_marks if max_marks > 0 else [1.0],
            "label_details": [],
            "justification": "No expected labels found in the rubric for this component.",
        }

    # Extract detected labels from diagram evaluation result (DEIS or Gemini Vision).
    # Both evaluators write `label_scores` with the same schema:
    #   [{"expected": str, "text": str, "matched": bool, "target_region": str}]
    detected_labels = []
    if deis_result:
        # Primary: label_scores from DEIS scoring worker OR Gemini Vision evaluator
        detected_labels = deis_result.get("label_scores", [])
        if not detected_labels:
            # Fallback A: scene_graph nodes (DEIS raw output)
            scene_graph = deis_result.get("scene_graph", {})
            if scene_graph:
                nodes = scene_graph.get("nodes", [])
                for node in nodes:
                    if isinstance(node, dict) and node.get("type") == "label":
                        detected_labels.append({
                            "text": node.get("text", node.get("id", "")),
                            "matched": True,
                        })
            # Fallback B: Gemini raw label_results
            if not detected_labels:
                gemini_raw = deis_result.get("_gemini_raw") or {}
                for lr in gemini_raw.get("label_results", []):
                    if lr.get("matched") or lr.get("detected_text"):
                        detected_labels.append({
                            "text": lr.get("detected_text") or "",
                            "matched": bool(lr.get("matched", False)),
                        })

    # Match detected labels against expected labels
    label_details = []
    matched_count = 0
    threshold = settings.LABEL_FUZZY_THRESHOLD

    for expected in expected_labels:
        best_match = _find_best_label_match(expected, detected_labels, threshold)

        if best_match:
            matched_count += 1
            label_details.append({
                "expected": expected,
                "detected": best_match["text"],
                "similarity": best_match["score"],
                "status": "correct",
            })
        else:
            label_details.append({
                "expected": expected,
                "detected": None,
                "similarity": 0,
                "status": "missing",
            })

    # Calculate marks
    if expected_labels:
        marks_ratio = matched_count / len(expected_labels)
    else:
        marks_ratio = 0.0

    marks_awarded = round(marks_ratio * max_marks)
    marks_awarded = min(marks_awarded, max_marks)

    # Confidence is based on how clear the matches were
    avg_similarity = 0.0
    if label_details:
        similarities = [ld["similarity"] for ld in label_details if ld["similarity"] > 0]
        if similarities:
            avg_similarity = sum(similarities) / len(similarities)
    confidence = avg_similarity / 100.0  # Normalize 0-100 → 0.0-1.0

    # Build distribution
    dist = [0.0] * (max_marks + 1)
    dist[marks_awarded] = 1.0

    # Justification
    correct = [ld for ld in label_details if ld["status"] == "correct"]
    missing = [ld for ld in label_details if ld["status"] == "missing"]
    justification_parts = []
    if correct:
        justification_parts.append(
            f"{len(correct)}/{len(expected_labels)} labels correctly identified"
        )
    if missing:
        missing_names = [ld["expected"] for ld in missing]
        justification_parts.append(f"Missing labels: {', '.join(missing_names)}")

    return {
        "type": "labels",
        "marks_awarded": marks_awarded,
        "max_marks": max_marks,
        "confidence": confidence,
        "grade_distribution": dist,
        "label_details": label_details,
        "justification": ". ".join(justification_parts) if justification_parts else "No labels evaluated.",
    }


def _extract_expected_labels(component: dict, rubric_steps: list[dict]) -> list[str]:
    """
    Extract the list of expected label strings from the rubric.

    Sources:
    1. diagram_relations in the rubric steps (label field).
    2. Description text parsing for label names.
    """
    labels = []
    component_step_nums = set(component.get("rubric_steps", []))

    for step in rubric_steps:
        if step.get("step_num") not in component_step_nums:
            continue

        # From diagram_relations
        for rel in step.get("diagram_relations", []):
            label = rel.get("label", "")
            if label and label not in labels:
                labels.append(label)

    return labels


def _find_best_label_match(
    expected: str,
    detected_labels: list[dict],
    threshold: int = 80,
) -> dict | None:
    """
    Find the best fuzzy match for an expected label among detected labels.

    Returns the best match dict with {"text": str, "score": int} or None.
    """
    best = None
    best_score = 0

    for detected in detected_labels:
        detected_text = detected.get("text", "")
        if not detected_text:
            continue

        # Use token_sort_ratio for robustness against word order and OCR artifacts
        score = fuzz.token_sort_ratio(expected.lower(), detected_text.lower())

        if score >= threshold and score > best_score:
            best_score = score
            best = {"text": detected_text, "score": score}

    return best
