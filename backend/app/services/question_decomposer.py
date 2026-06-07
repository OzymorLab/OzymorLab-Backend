"""
Question Decomposer — Question Intent & Rubric Decomposition Engine.

Before any grading begins, this module determines what evaluation components
are expected for each question. It uses a hybrid approach:
  - If the rubric steps already have `component_type` tags → use those (teacher intent).
  - Otherwise → call Gemini to auto-decompose the question into components.

This creates dynamic evaluation workflows:
  "Explain with diagram" → text + diagram + labels
  "Draw circuit"         → diagram only
  "Define law"           → text only
  "Derive equation"      → reasoning + text
  "Label map"            → diagram + labels
"""
import logging
from app.services.llm_client import call_llm, parse_json_response

logger = logging.getLogger(__name__)

# ── Supported component types ──
COMPONENT_TYPES = {"text", "diagram", "labels", "reasoning"}

DECOMPOSITION_SYSTEM_PROMPT = """You are an expert educational assessment architect.
Given a question and its rubric steps, decompose the expected answer into independent evaluation components.

Each component must be one of: text, diagram, labels, reasoning.

- "text": Explanations, definitions, theoretical reasoning, concept correctness.
- "diagram": Structural diagrams, circuit drawings, biological drawings, maps.
- "labels": Named labels on diagrams, terminology, annotations.
- "reasoning": Step-by-step derivations, mathematical proofs, procedural logic, presentation.

Output ONLY valid JSON — no preamble, no markdown, no code fences."""


def decompose_from_rubric(rubric_steps: list[dict]) -> list[dict]:
    """
    Extract evaluation components directly from rubric steps that already
    have `component_type` tags (teacher-defined decomposition).

    Groups rubric steps by their component_type and creates one component
    per group with aggregated marks.
    """
    components_map: dict[str, dict] = {}

    for step in rubric_steps:
        ctype = step.get("component_type", "").lower()
        if ctype not in COMPONENT_TYPES:
            continue

        if ctype not in components_map:
            components_map[ctype] = {
                "type": ctype,
                "description": "",
                "max_marks": 0,
                "rubric_steps": [],
                "source": "rubric_tag",
            }

        components_map[ctype]["max_marks"] += step.get("marks", 0)
        components_map[ctype]["rubric_steps"].append(step.get("step_num", 0))
        desc = step.get("description", "")
        if desc:
            if components_map[ctype]["description"]:
                components_map[ctype]["description"] += "; " + desc
            else:
                components_map[ctype]["description"] = desc

    return list(components_map.values())


def decompose_via_llm(question_text: str, rubric_steps: list[dict]) -> list[dict]:
    """
    Use Gemini to auto-decompose a question into evaluation components.
    Used when rubric steps don't have explicit component_type tags.
    """
    steps_desc = "\n".join(
        f"  Step {s.get('step_num', i+1)} ({s.get('marks', 0)} marks): "
        f"{s.get('description', '')} [type: {s.get('step_type', 'statement')}]"
        for i, s in enumerate(rubric_steps)
    )

    prompt = f"""QUESTION:
{question_text}

RUBRIC STEPS:
{steps_desc}

Decompose the expected answer into evaluation components.
For each component, specify which rubric step numbers belong to it.

Return JSON array:
[
  {{
    "type": "<text|diagram|labels|reasoning>",
    "description": "<what this component evaluates>",
    "max_marks": <sum of marks for the rubric steps in this component>,
    "rubric_steps": [<list of step_num integers>]
  }}
]

Rules:
- Every rubric step must appear in exactly one component.
- Merge related steps into the same component.
- If a step says "diagram" or "draw", it belongs to the "diagram" component.
- If a step mentions "label" or "annotate", it belongs to the "labels" component.
- If a step involves derivation, formula, or proof, it belongs to "reasoning".
- All other explanation/definition/theory steps belong to "text".
"""

    result = call_llm(
        prompt,
        system_prompt=DECOMPOSITION_SYSTEM_PROMPT,
        temperature=0.0,
        call_type="question_decomposition",
    )

    if not result["success"]:
        logger.error(f"LLM decomposition failed: {result.get('error')}")
        return _fallback_decomposition(rubric_steps)

    parsed = parse_json_response(result["response_text"])
    if not parsed or not isinstance(parsed, list):
        logger.warning("LLM decomposition returned unparseable result. Using fallback.")
        return _fallback_decomposition(rubric_steps)

    # Validate component types
    for comp in parsed:
        if comp.get("type") not in COMPONENT_TYPES:
            comp["type"] = "text"
        comp["source"] = "llm_decomposition"

    return parsed


def _fallback_decomposition(rubric_steps: list[dict]) -> list[dict]:
    """
    Fallback: group rubric steps by their existing step_type field.
    Maps step_type → component_type:
      statement → text, derivation → reasoning, result → text, diagram → diagram
    """
    type_map = {
        "statement": "text",
        "derivation": "reasoning",
        "result": "text",
        "diagram": "diagram",
    }

    components_map: dict[str, dict] = {}
    for step in rubric_steps:
        step_type = step.get("step_type", "statement")
        ctype = type_map.get(step_type, "text")

        if ctype not in components_map:
            components_map[ctype] = {
                "type": ctype,
                "description": "",
                "max_marks": 0,
                "rubric_steps": [],
                "source": "fallback",
            }

        components_map[ctype]["max_marks"] += step.get("marks", 0)
        components_map[ctype]["rubric_steps"].append(step.get("step_num", 0))

    return list(components_map.values())


def decompose_question(
    rubric_steps: list[dict],
    question_text: str = "",
) -> list[dict]:
    """
    Main entry point. Decomposes a question into evaluation components.

    Strategy (Option C from the plan):
      1. If rubric steps have component_type tags → use those.
      2. Otherwise → call Gemini to auto-decompose.
      3. If Gemini fails → fallback to step_type grouping.

    Args:
        rubric_steps: List of rubric step dicts from the task rubric.
        question_text: The original question text (needed for LLM decomposition).

    Returns:
        List of component dicts, each with:
            type, description, max_marks, rubric_steps, source
    """
    # Check if any rubric steps have explicit component_type tags
    tagged_steps = [s for s in rubric_steps if s.get("component_type") in COMPONENT_TYPES]

    if len(tagged_steps) == len(rubric_steps) and len(tagged_steps) > 0:
        # All steps are tagged — use teacher-defined decomposition
        logger.info("Using teacher-defined component decomposition from rubric tags.")
        return decompose_from_rubric(rubric_steps)

    if question_text:
        # Use LLM to auto-decompose
        logger.info("Using LLM-powered question decomposition.")
        return decompose_via_llm(question_text, rubric_steps)

    # No question text and no tags — use fallback
    logger.info("Using fallback step_type-based decomposition.")
    return _fallback_decomposition(rubric_steps)
