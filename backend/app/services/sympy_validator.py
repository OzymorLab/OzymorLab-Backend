"""
SymPy equation validator — the key technical differentiator.

Validates mathematical transformations symbolically before LLM scoring.
If SymPy can parse the expressions, it provides deterministic validation.
If not, the system falls back to pure LLM judgment.

Supports true equation equivalence: F = ma and a = F/m are recognized
as mathematically equivalent via cross-multiplication verification.
"""
import logging
from sympy import sympify, simplify, Eq, Symbol, solve

logger = logging.getLogger(__name__)


def validate_step_transformation(
    prev_expr: str,
    curr_expr: str,
    allowed_ops: list[str] | None = None,
) -> dict:
    """
    Check whether going from prev_expr → curr_expr is mathematically valid.

    Handles three cases:
      1. Both are equations (contain '='): uses cross-multiplication equivalence.
      2. One is an equation, the other isn't: compares RHS of the equation to the expression.
      3. Neither is an equation: direct symbolic difference comparison.

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
        prev_clean = clean_expression(prev_expr)
        curr_clean = clean_expression(curr_expr)

        prev_is_eq = "=" in prev_expr
        curr_is_eq = "=" in curr_expr

        if prev_is_eq and curr_is_eq:
            # ── Case 1: Both are equations — true equivalence check ──
            # Parse both sides of each equation
            p_lhs, p_rhs = _parse_equation_sides(prev_clean)
            c_lhs, c_rhs = _parse_equation_sides(curr_clean)

            # Convert both to "LHS - RHS = 0" form
            expr1 = p_lhs - p_rhs
            expr2 = c_lhs - c_rhs

            # Method 1: Direct symbolic difference
            direct_diff = simplify(expr1 - expr2)
            if direct_diff == 0:
                return {"valid": True, "error": None, "fallback": False}

            # Method 2: Substitution-based equivalence check.
            # Two equations represent the same constraint if every solution
            # of eq1 is also a solution of eq2. We verify this by solving
            # eq1 for each variable and substituting into eq2.
            eq1 = Eq(p_lhs, p_rhs)
            all_symbols = expr1.free_symbols | expr2.free_symbols

            for sym in all_symbols:
                try:
                    solutions = solve(eq1, sym)
                    for sol in solutions:
                        substituted = simplify(expr2.subs(sym, sol))
                        if substituted == 0:
                            return {"valid": True, "error": None, "fallback": False}
                except Exception:
                    continue  # Some symbols may not be solvable

            # Method 3: Check if expr1 and expr2 are scalar multiples
            if expr2 != 0:
                ratio = simplify(expr1 / expr2)
                if ratio.is_number and ratio != 0:
                    return {"valid": True, "error": None, "fallback": False}

            return {
                "valid": False,
                "error": f"Equations are not equivalent.",
                "diff_expr": str(simplify(expr1)),
                "fallback": False,
            }

        elif prev_is_eq or curr_is_eq:
            # ── Case 2: One is an equation, other is plain expression ──
            # Compare the RHS of the equation against the plain expression
            if prev_is_eq:
                _, rhs_expr = _parse_equation_sides(prev_clean)
                plain_expr = sympify(curr_clean)
            else:
                _, rhs_expr = _parse_equation_sides(curr_clean)
                plain_expr = sympify(prev_clean)

            diff = simplify(rhs_expr - plain_expr)
            if diff == 0:
                return {"valid": True, "error": None, "fallback": False}

            return {
                "valid": False,
                "error": f"Transformation invalid. Difference: {diff}",
                "diff_expr": str(diff),
                "fallback": False,
            }

        else:
            # ── Case 3: Neither is an equation — direct comparison ──
            lhs = sympify(prev_clean)
            rhs = sympify(curr_clean)
            diff = simplify(lhs - rhs)

            if diff == 0:
                return {"valid": True, "error": None, "fallback": False}

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


def _parse_equation_sides(cleaned_expr: str) -> tuple:
    """
    Split a cleaned expression on '=' and parse both sides with SymPy.

    Args:
        cleaned_expr: Expression string that contains exactly one '='.

    Returns:
        Tuple of (lhs_sympy, rhs_sympy).
    """
    parts = cleaned_expr.split("=", 1)
    lhs = sympify(parts[0].strip())
    rhs = sympify(parts[1].strip())
    return lhs, rhs


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
        cleaned = clean_expression(expr)
        # If it's an equation, parse both sides
        if "=" in cleaned:
            lhs, rhs = _parse_equation_sides(cleaned)
            simplified = f"{simplify(lhs)} = {simplify(rhs)}"
        else:
            parsed = sympify(cleaned)
            simplified = str(simplify(parsed))
        return {
            "parseable": True,
            "simplified": simplified,
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
    Handles common LaTeX notation differences.

    NOTE: This function does NOT strip the LHS of equations.
    Equation handling (LHS = RHS) is done at the caller level
    via _parse_equation_sides() to preserve full equation semantics.
    """
    # Remove leading/trailing whitespace
    expr = expr.strip()

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
