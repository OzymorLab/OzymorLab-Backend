"""
SymPy equation validator — the key technical differentiator.

Validates mathematical transformations symbolically before LLM scoring.
If SymPy can parse the expressions, it provides deterministic validation.
If not, the system falls back to pure LLM judgment.
"""
import logging
from sympy import sympify, simplify

logger = logging.getLogger(__name__)


def validate_step_transformation(
    prev_expr: str,
    curr_expr: str,
    allowed_ops: list[str] | None = None,
) -> dict:
    """
    Check whether going from prev_expr → curr_expr is mathematically valid.

    Args:
        prev_expr: The previous expression (e.g., "F = ma")
        curr_expr: The current expression (e.g., "a = F/m")
        allowed_ops: Optional list of allowed operations (unused in MVP)

    Returns:
        dict with keys:
        - valid: True if transformation is correct, False if wrong, None if unparseable
        - error: Error description if invalid
        - diff_expr: Simplified difference if invalid
        - fallback: True if SymPy couldn't parse (LLM should judge)
    """
    try:
        # Clean up expressions for SymPy parsing
        lhs = sympify(clean_expression(prev_expr))
        rhs = sympify(clean_expression(curr_expr))
        diff = simplify(lhs - rhs)

        if diff == 0:
            return {"valid": True, "error": None, "fallback": False}

        # Non-zero diff: transformation introduced an error
        return {
            "valid": False,
            "error": f"Transformation invalid. Difference: {diff}",
            "diff_expr": str(diff),
            "fallback": False,
        }
    except Exception as e:
        # Parse failure = LLM fallback for this step
        logger.debug(f"SymPy parse failed for '{prev_expr}' → '{curr_expr}': {e}")
        return {
            "valid": None,
            "error": str(e),
            "fallback": True,
        }


def validate_expression(expr: str) -> dict:
    """
    Validate a single expression — check if SymPy can parse it.

    Returns:
        dict with keys:
        - parseable: bool
        - simplified: str (simplified form if parseable)
        - error: str | None
    """
    try:
        parsed = sympify(clean_expression(expr))
        simplified = simplify(parsed)
        return {
            "parseable": True,
            "simplified": str(simplified),
            "error": None,
        }
    except Exception as e:
        return {
            "parseable": False,
            "simplified": None,
            "error": str(e),
        }


def validate_expected_against_student(
    expected_exprs: list[str],
    student_equations: list[str],
) -> list[dict]:
    """
    Compare student equations against expected expressions from the rubric.

    Returns a list of validation results, one per expected expression.
    """
    results = []

    for expected in expected_exprs:
        best_match = None
        best_result = {"valid": None, "fallback": True, "error": "No matching student equation found"}

        for student_eq in student_equations:
            result = validate_step_transformation(expected, student_eq)
            if result["valid"] is True:
                best_match = student_eq
                best_result = result
                break
            elif result["valid"] is False and best_match is None:
                best_match = student_eq
                best_result = result

        results.append({
            "expected": expected,
            "student_match": best_match,
            **best_result,
        })

    return results


def clean_expression(expr: str) -> str:
    """
    Clean an expression string for SymPy parsing.
    Handles common notation differences.
    """
    # Remove leading/trailing whitespace
    expr = expr.strip()

    # Remove equation labels like "F = " at the start (keep only the RHS)
    # But only if there's a clear "variable = expression" pattern
    if "=" in expr:
        parts = expr.split("=", 1)
        if len(parts) == 2:
            # Return the full equation as SymPy Eq would need it
            # For simple comparison, just return one side
            expr = parts[1].strip()

    # Replace common LaTeX notation with SymPy-compatible forms
    expr = expr.replace("\\cdot", "*")
    expr = expr.replace("\\times", "*")
    expr = expr.replace("\\div", "/")
    expr = expr.replace("\\frac{", "(")
    expr = expr.replace("}{", ")/(")
    expr = expr.replace("}", ")")
    expr = expr.replace("{", "(")
    expr = expr.replace("^", "**")

    return expr
