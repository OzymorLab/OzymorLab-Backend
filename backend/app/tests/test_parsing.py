"""
Unit tests for Stateful Parsing Pipeline — contextual breadcrumbs tracking and segment boundaries.
"""
import pytest
from app.services.parsing import (
    AnswerSheetState,
    segment_into_steps,
    extract_equations,
    classify_step_type,
    compute_parse_confidence,
)


def test_answer_sheet_state_breadcrumbs():
    """Verify that stateful tracker accurately parses and inherits contextual breadcrumbs."""
    tracker = AnswerSheetState()

    # Base state
    assert tracker.get_state_string() == "Unknown Context"

    # Match Section
    mutated = tracker.update_from_text("Section A")
    assert mutated is True
    assert tracker.get_state_string() == "Sec A"

    # Match Question
    mutated = tracker.update_from_text("Question 2")
    assert mutated is True
    assert tracker.get_state_string() == "Sec A | Q2"

    # Match Subquestion (bare notation, inherits Q2)
    mutated = tracker.update_from_text("(b) Determine the velocity.")
    assert mutated is True
    assert tracker.get_state_string() == "Sec A | Q2(B)"

    # Match new section reset
    mutated = tracker.update_from_text("Part C")
    assert mutated is True
    assert tracker.get_state_string() == "Sec C"


def test_segment_into_steps():
    """Verify that raw text is segmented into discrete structured steps with contexts."""
    raw_text = """
    Section B
    Question 4
    Step 1: Write down given values.
    We are given mass m = 10 kg.
    Step 2: Apply Newton's Second Law.
    F = m * a
    Therefore, the force is 50 N.
    """

    steps = segment_into_steps(raw_text)

    # Verify segmentation count
    assert len(steps) >= 3

    # Verify step text contents
    step_texts = [s["text"] for s in steps]
    assert any("Section B" in text for text in step_texts)
    assert any("Question 4" in text for text in step_texts)
    assert any("Step 1" in text or "mass m" in text for text in step_texts)
    assert any("Step 2" in text or "F = m" in text for text in step_texts)


def test_extract_equations():
    """Verify LaTeX math extraction."""
    text = "Let the formula be $E = m * c^2$ and $F = m * a$."
    eqs = extract_equations(text)
    assert len(eqs) == 2
    assert "E = m * c^2" in eqs
    assert "F = m * a" in eqs


def test_classify_step_type():
    """Verify classification of steps based on heuristics."""
    assert classify_step_type("Therefore the answer is 12.") == "result"
    assert classify_step_type("Let's integrate the function x.") == "derivation"
    assert classify_step_type("Refer to Figure 1 diagram.") == "diagram"
    assert classify_step_type("This is a simple descriptive sentence.") == "statement"


def test_compute_parse_confidence():
    """Verify parse confidence bounds and score calculation."""
    # Empty raw text -> 0.0
    assert compute_parse_confidence("", []) == 0.0

    # Structured steps and equations increase confidence
    mock_steps = [
        {"equations": ["F = ma"]},
        {"equations": ["a = F/m"]},
        {"equations": ["m = 10"]}
    ]
    raw_text = "Step 1: F = ma. Step 2: a = F/m. Step 3: m = 10. Done!"
    conf = compute_parse_confidence(raw_text, mock_steps)
    assert conf >= 0.7
