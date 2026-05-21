"""
Unit tests for SymPy equation validator — the key technical differentiator.
"""
import pytest
from app.services.sympy_validator import (
    validate_step_transformation,
    validate_expression,
    validate_expected_against_student,
    clean_expression,
)


def test_clean_expression():
    """Verify LaTeX math notation is cleaned into SymPy friendly formats."""
    assert clean_expression("a \\cdot b") == "a * b"
    assert clean_expression("a \\times b") == "a * b"
    assert clean_expression("a \\div b") == "a / b"
    assert clean_expression("\\frac{x}{y}") == "(x)/(y)"
    assert clean_expression("x^2") == "x**2"
    assert clean_expression("  x + y  ") == "x + y"


def test_validate_expression():
    """Verify that validate_expression correctly assesses SymPy parseability."""
    # Parseable plain expressions
    res = validate_expression("x**2 + y")
    assert res["parseable"] is True
    assert res["simplified"] == "x**2 + y"

    # Parseable equations
    res = validate_expression("F = m * a")
    assert res["parseable"] is True
    assert "F = a*m" in res["simplified"] or "F = m*a" in res["simplified"]

    # Unparseable syntax
    res = validate_expression("x +=/ 3")
    assert res["parseable"] is False
    assert res["error"] is not None


def test_validate_step_transformation_equations():
    """Test equation equivalence checks (Case 1: Both are equations)."""
    # Simple transpositions
    res = validate_step_transformation("F = m * a", "a = F / m")
    assert res["valid"] is True
    assert res["fallback"] is False

    res = validate_step_transformation("F = m * a", "m = F / a")
    assert res["valid"] is True

    # Scalar multiples
    res = validate_step_transformation("x + y = 5", "2 * x + 2 * y = 10")
    assert res["valid"] is True

    # Mathematically invalid transpositions
    res = validate_step_transformation("F = m * a", "a = F * m")
    assert res["valid"] is False
    assert res["fallback"] is False


def test_validate_step_transformation_mixed():
    """Test mixed validation checks (Case 2: One is an equation, other is plain expression)."""
    # Correct substitution/equivalence to RHS
    res = validate_step_transformation("E = m * c**2", "m * c**2")
    assert res["valid"] is True

    # Incorrect
    res = validate_step_transformation("E = m * c**2", "m * c")
    assert res["valid"] is False


def test_validate_step_transformation_plain():
    """Test plain expression checks (Case 3: Neither is an equation)."""
    res = validate_step_transformation("x + x", "2 * x")
    assert res["valid"] is True

    res = validate_step_transformation("(x + y)**2", "x**2 + 2*x*y + y**2")
    assert res["valid"] is True

    res = validate_step_transformation("x + 1", "x + 2")
    assert res["valid"] is False


def test_validate_step_transformation_fallback():
    """Verify that unparseable text returns fallback=True gracefully without crashing."""
    res = validate_step_transformation("The force is proportional to acceleration", "F = ma")
    assert res["valid"] is None
    assert res["fallback"] is True
    assert res["error"] is not None


def test_validate_expected_against_student():
    """Test comparison of multiple expected expressions against student work."""
    expected = ["F = m * a", "E = m * c**2"]
    student = ["a = F / m", "x = y + 2", "E = c**2 * m"]

    results = validate_expected_against_student(expected, student)
    assert len(results) == 2

    # First expected "F = m * a" matched "a = F / m"
    assert results[0]["expected"] == "F = m * a"
    assert results[0]["student_match"] == "a = F / m"
    assert results[0]["valid"] is True

    # Second expected "E = m * c**2" matched "E = c**2 * m"
    assert results[1]["expected"] == "E = m * c**2"
    assert results[1]["student_match"] == "E = c**2 * m"
    assert results[1]["valid"] is True
