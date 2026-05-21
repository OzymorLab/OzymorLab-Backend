"""
Unit tests for Score Fusion Engine — convolved uncertainties and joint evaluation fusion.
"""
import pytest
from app.services.score_fusion import (
    fuse_component_scores,
    compute_confidence,
    _convolve_distributions,
)


def test_fuse_component_scores_empty():
    """Verify fuse_component_scores handles empty input gracefully."""
    res = fuse_component_scores([])
    assert res["grade"] == 0
    assert res["max_grade"] == 0
    assert res["confidence"] == 0.0
    assert res["grade_distribution"] == [1.0]
    assert res["component_grades"] == []
    assert res["weakest_component"] is None


def test_fuse_component_scores_valid():
    """Verify core fusion math, convolved distributions, and weakest components."""
    components = [
        {
            "type": "text",
            "marks_awarded": 4,
            "max_marks": 5,
            "confidence": 0.9,
            "grade_distribution": [0.0, 0.0, 0.0, 0.1, 0.8, 0.1],  # sharp peak at 4
            "justification": "Matches expected physics definition.",
        },
        {
            "type": "diagram",
            "marks_awarded": 3,
            "max_marks": 5,
            "confidence": 0.6,
            "grade_distribution": [0.0, 0.0, 0.1, 0.6, 0.2, 0.1],  # peak at 3
            "justification": "Diagram drawn but slightly misaligned labels.",
        }
    ]

    res = fuse_component_scores(components)

    # 1. Marks math
    assert res["grade"] == 7
    assert res["max_grade"] == 10

    # 2. Weighted confidence: 0.9 * (5/10) + 0.6 * (5/10) = 0.75
    assert res["confidence"] == 0.75

    # 3. Weakest component detection
    assert res["weakest_component"]["type"] == "diagram"
    assert res["weakest_component"]["confidence"] == 0.6

    # 4. Joint convolved distribution check
    convolved = res["grade_distribution"]
    assert len(convolved) > 0
    # Probabilities should sum to 1.0
    assert pytest.approx(sum(convolved), 1e-5) == 1.0

    # 5. Justification check
    assert "[TEXT] 4/5" in res["justification"]
    assert "[DIAGRAM] 3/5" in res["justification"]


def test_convolve_distributions():
    """Verify that _convolve_distributions convolves distributions cleanly."""
    d1 = [0.1, 0.8, 0.1]
    d2 = [0.2, 0.7, 0.1]

    res = _convolve_distributions([d1, d2])
    # The convolved size of [3] and [3] is 3 + 3 - 1 = 5
    assert len(res) == 5
    assert pytest.approx(sum(res), 1e-5) == 1.0


def test_compute_confidence():
    """Verify confidence entropy-based calculation."""
    # Sharp single-point distribution has 0 entropy -> 1.0 confidence
    assert compute_confidence([1.0]) == 1.0
    assert compute_confidence([0.0, 1.0, 0.0]) == 1.0

    # High entropy (flat distribution) has lower confidence
    flat = [0.25, 0.25, 0.25, 0.25]
    conf = compute_confidence(flat)
    assert conf < 0.1  # very close to 0
