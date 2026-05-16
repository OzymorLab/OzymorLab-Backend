"""
Score Fusion Engine — Merges independent component evaluation results into a final grade.

Takes the results from all parallel pipelines (text, diagram, labels, reasoning)
and produces a single cumulative grade with:
  - combined marks (sum of component marks)
  - overall confidence (weighted average)
  - combined grade distribution (convolution)
  - component breakdown for explainability
"""
import logging
import numpy as np
from scipy.signal import fftconvolve

logger = logging.getLogger(__name__)


def fuse_component_scores(component_results: list[dict]) -> dict:
    """
    Merge independent component scores into a final cumulative grade.

    Each component_result should have:
        - type: "text" | "diagram" | "labels" | "reasoning"
        - marks_awarded: int
        - max_marks: int
        - confidence: float (0.0 - 1.0)
        - grade_distribution: list[float]
        - justification: str

    Returns:
        Fused result dict with:
            - grade: total marks
            - max_grade: total possible marks
            - confidence: weighted overall confidence
            - grade_distribution: convolved distribution
            - component_grades: list of per-component results
            - justification: combined justification
            - weakest_component: the component with lowest confidence
    """
    if not component_results:
        return {
            "grade": 0,
            "max_grade": 0,
            "confidence": 0.0,
            "grade_distribution": [1.0],
            "component_grades": [],
            "justification": "No evaluation components found.",
            "weakest_component": None,
        }

    # Calculate cumulative totals
    total_grade = sum(cr["marks_awarded"] for cr in component_results)
    max_grade = sum(cr["max_marks"] for cr in component_results)

    # Weighted confidence (weighted by max_marks proportion)
    if max_grade > 0:
        weighted_confidence = sum(
            cr["confidence"] * (cr["max_marks"] / max_grade)
            for cr in component_results
        )
    else:
        weighted_confidence = 0.0

    # Convolve grade distributions from all components
    distributions = [
        cr["grade_distribution"]
        for cr in component_results
        if cr.get("grade_distribution")
    ]
    total_distribution = _convolve_distributions(distributions)

    # Find weakest component (lowest confidence)
    weakest = min(component_results, key=lambda cr: cr.get("confidence", 0.0))

    # Build combined justification
    justification_parts = []
    for cr in component_results:
        ctype = cr.get("type", "unknown").upper()
        marks = cr.get("marks_awarded", 0)
        max_m = cr.get("max_marks", 0)
        conf = cr.get("confidence", 0.0)
        justification_parts.append(
            f"[{ctype}] {marks}/{max_m} (confidence={conf:.2f}): {cr.get('justification', '')}"
        )

    return {
        "grade": total_grade,
        "max_grade": max_grade,
        "confidence": round(weighted_confidence, 4),
        "grade_distribution": total_distribution,
        "component_grades": component_results,
        "justification": " | ".join(justification_parts),
        "weakest_component": {
            "type": weakest.get("type"),
            "confidence": weakest.get("confidence", 0.0),
        },
    }


def _convolve_distributions(distributions: list[list[float]]) -> list[float]:
    """
    Convolve multiple grade distributions into a single joint distribution.
    This is the mathematically precise representation of combined uncertainty.
    """
    if not distributions:
        return [1.0]

    result = np.array(distributions[0], dtype=float)
    for dist in distributions[1:]:
        result = fftconvolve(result, np.array(dist, dtype=float), mode="full")

    # Normalize
    total = result.sum()
    if total > 0:
        result = result / total

    return result.tolist()


def compute_confidence(grade_distribution: list[float]) -> float:
    """Compute confidence as 1 - normalized entropy of the distribution."""
    dist = np.array(grade_distribution, dtype=float)
    dist = dist[dist > 0]
    if len(dist) <= 1:
        return 1.0
    entropy = -np.sum(dist * np.log2(dist))
    max_entropy = np.log2(len(grade_distribution))
    if max_entropy == 0:
        return 1.0
    return float(1.0 - (entropy / max_entropy))
