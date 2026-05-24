"""
Question Paper Service — AI-powered rubric generation from uploaded question papers.

Flow:
  1. Teacher uploads a question paper (PDF/image).
  2. Text is extracted using PyMuPDF or Gemini Vision.
  3. Full text is sent to Gemini with a structured prompt to decompose into RubricSteps.
  4. Returns a draft rubric for teacher review/edit before confirmation.
"""
import logging
from app.services.parsing import extract_text
from app.services.llm_client import call_gemini, parse_json_response

logger = logging.getLogger(__name__)


RUBRIC_GENERATION_SYSTEM_PROMPT = """You are an expert educational assessment architect for the Indian board examination system (CBSE, ICSE, State Boards).

Given the full text of a question paper, you must:
1. Identify every individual question and sub-question.
2. For each question, produce a structured rubric step with marking criteria.
3. Infer the expected answer type (text, diagram, labels, reasoning) from the question phrasing.
4. Allocate marks based on explicit marks mentioned in the paper, or infer proportional marks if not stated.

Output ONLY valid JSON — no preamble, no markdown, no code fences.

IMPORTANT RULES:
- "State", "Define", "Explain", "Describe" → component_type: "text"
- "Draw", "Sketch", "Diagram" → component_type: "diagram"  
- "Label", "Annotate", "Mark on the diagram" → component_type: "labels"
- "Derive", "Prove", "Show that", "Calculate" → component_type: "reasoning"
- If a question says "Draw and label" → create TWO steps: one diagram + one labels
- If marks are explicitly stated like "[3 marks]" or "(3)", use those values
- step_num should be globally sequential (1, 2, 3, ...) across all sections
- Group sub-questions under their parent question in the description"""


def extract_question_paper_text(file_data: bytes, file_type: str) -> str:
    """Extract all text from a question paper PDF or image."""
    return extract_text(file_data, file_type)


def generate_rubric_from_text(
    question_text: str,
    subject: str = "General",
    board: str = "CBSE",
    grade_level: str = "Class 12",
    max_marks: int = 100,
) -> dict:
    """
    Use Gemini to auto-generate a structured rubric from question paper text.

    Args:
        question_text: Full extracted text of the question paper.
        subject: Subject name (Physics, Chemistry, etc.)
        board: Board name (CBSE, ICSE, State)
        grade_level: Grade level (Class 10, Class 12)
        max_marks: Total marks for the paper.

    Returns:
        Dict with:
            - steps: List of RubricStep-compatible dicts
            - grading_notes: Auto-generated grading guidance
            - ai_confidence: Estimated confidence (0.0 - 1.0)
    """
    prompt = f"""QUESTION PAPER TEXT:
\"\"\"
{question_text}
\"\"\"

METADATA:
- Subject: {subject}
- Board: {board}
- Grade Level: {grade_level}
- Total Marks: {max_marks}

Decompose this question paper into individual rubric steps. Each step represents one question or sub-question.

CRITICAL SUBJECT FILTER:
- Only decompose questions that belong to the specified Subject: {subject}.
- If the question paper contains other subjects (e.g. Chemistry, Mathematics, Biology), completely ignore sections and questions belonging to them.
- Ensure the sum of the marks of all steps matches the Total Marks ({max_marks}) specified in the metadata. Scale the marks proportionally if the original marks sum to a different total.

Return a JSON object:
{{
  "steps": [
    {{
      "step_num": 1,
      "description": "<the exact question text or a clear summary>",
      "marks": <integer>,
      "step_type": "<statement|derivation|result|diagram>",
      "component_type": "<text|diagram|labels|reasoning>",
      "expected_exprs": ["<SymPy-parseable equation if applicable, else empty list>"],
      "marking_notes": "<specific marking criteria or accepted alternatives>",
      "partial_credit": true,
      "diagram_relations": []
    }}
  ],
  "grading_notes": "<overall grading guidance for this paper>",
  "total_marks_allocated": <sum of all step marks>,
  "question_count": <number of distinct questions identified>
}}

Rules:
- Every question/sub-question relevant to {subject} must have its own step.
- Marks MUST sum to {max_marks}.
- If marks are not explicitly stated in the paper, distribute them proportionally.
- For "Draw and label" questions, create separate diagram and labels steps.
- Include expected_exprs for derivation/calculation questions where you can infer the expected formula.
- Use marking_notes to specify what earns partial credit.
"""

    result = call_gemini(
        prompt,
        system_prompt=RUBRIC_GENERATION_SYSTEM_PROMPT,
        temperature=0.1,
        max_tokens=8192,
        call_type="rubric_generation",
        response_mime_type="application/json",
    )

    if not result["success"]:
        logger.error(f"Rubric generation failed: {result.get('error')}")
        return _fallback_rubric(question_text, max_marks)

    parsed = parse_json_response(result["response_text"])
    if not parsed or not isinstance(parsed, dict) or "steps" not in parsed:
        logger.warning("Rubric generation returned unparseable result. Using fallback.")
        return _fallback_rubric(question_text, max_marks)

    # Validate and clean up steps
    steps = parsed.get("steps", [])
    valid_component_types = {"text", "diagram", "labels", "reasoning"}
    valid_step_types = {"statement", "derivation", "result", "diagram"}

    for i, step in enumerate(steps):
        # Ensure required fields exist
        step.setdefault("step_num", i + 1)
        step.setdefault("description", f"Question {i + 1}")
        step.setdefault("marks", 1)
        step.setdefault("step_type", "statement")
        step.setdefault("component_type", "text")
        step.setdefault("expected_exprs", [])
        step.setdefault("marking_notes", "")
        step.setdefault("partial_credit", True)
        step.setdefault("diagram_relations", [])

        # Validate enums
        if step["component_type"] not in valid_component_types:
            step["component_type"] = "text"
        if step["step_type"] not in valid_step_types:
            step["step_type"] = "statement"

        # Ensure marks is a positive integer
        step["marks"] = max(int(step["marks"]), 0)

    # Compute confidence based on quality indicators
    total_allocated = sum(s["marks"] for s in steps)
    marks_accuracy = 1.0 - abs(total_allocated - max_marks) / max(max_marks, 1)
    has_variety = len(set(s["component_type"] for s in steps)) > 1
    confidence = min(
        0.5 + (marks_accuracy * 0.3) + (0.1 if has_variety else 0) + (0.1 if len(steps) > 3 else 0),
        1.0,
    )

    return {
        "steps": steps,
        "grading_notes": parsed.get("grading_notes", f"Auto-generated rubric for {subject} ({board})."),
        "ai_confidence": round(confidence, 2),
        "total_marks_allocated": total_allocated,
        "question_count": parsed.get("question_count", len(steps)),
    }


def _fallback_rubric(question_text: str, max_marks: int) -> dict:
    """
    Fallback when Gemini fails: create a single catch-all rubric step.
    """
    return {
        "steps": [
            {
                "step_num": 1,
                "description": "Full answer evaluation (auto-rubric generation failed — please edit manually)",
                "marks": max_marks,
                "step_type": "statement",
                "component_type": "text",
                "expected_exprs": [],
                "marking_notes": "AI could not decompose this question paper. Please edit this rubric manually.",
                "partial_credit": True,
                "diagram_relations": [],
            }
        ],
        "grading_notes": "Fallback rubric — AI decomposition failed. Please review and edit.",
        "ai_confidence": 0.1,
        "total_marks_allocated": max_marks,
        "question_count": 1,
    }


def process_question_paper(
    file_data: bytes,
    file_type: str,
    subject: str = "General",
    board: str = "CBSE",
    grade_level: str = "Class 12",
    max_marks: int = 100,
) -> dict:
    """
    Full pipeline: extract text from question paper → generate rubric.

    Returns:
        Dict with extracted_text, draft_rubric, and ai_confidence.
    """
    # Step 1: Extract text
    logger.info(f"Extracting text from question paper ({file_type})...")
    extracted_text = extract_question_paper_text(file_data, file_type)

    if not extracted_text or not extracted_text.strip():
        logger.warning("No text extracted from question paper.")
        return {
            "extracted_text": "",
            "draft_rubric": _fallback_rubric("", max_marks),
            "ai_confidence": 0.0,
        }

    # Step 2: Generate rubric from extracted text
    logger.info(f"Generating rubric from extracted text ({len(extracted_text)} chars)...")
    rubric_result = generate_rubric_from_text(
        extracted_text, subject, board, grade_level, max_marks
    )

    return {
        "extracted_text": extracted_text,
        "draft_rubric": rubric_result,
        "ai_confidence": rubric_result.get("ai_confidence", 0.0),
    }
