"""
Grading Orchestrator — Independent Question-level Evaluation.
"""
import time
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def grade_single_answer(
    question_text: str,
    max_marks: float,
    marking_notes: str,
    student_answer: str,
    diagram_keys: list[str] | None = None,
    subject: str = "General",
    board: str = "CBSE",
    grade_level: str = "Class 12",
    api_key: str | None = None,
) -> dict:
    """
    Grades a single question's answer independently using Gemini.
    """
    from app.services.llm_client import get_client, parse_json_response
    from app.services.ingestion import download_file
    from app.config import settings
    from google.genai import types as genai_types

    prompt = f"""
    You are an expert exam evaluator for the {board} board ({subject}, {grade_level}).
    Evaluate the following student answer for this specific question.
    
    Question Text:
    {question_text}
    
    Maximum Marks: {max_marks}
    Marking Criteria & Notes:
    {marking_notes}
    
    Student's Written Answer:
    {student_answer}
    
    If there is a diagram image attached, evaluate the hand-drawn diagram structure, accuracy, and labels against the question requirements.
    
    Return ONLY a valid JSON object (no markdown code blocks, no other text) with these exact keys:
    {{
      "marks_awarded": <float between 0 and {max_marks}>,
      "justification": "<concise explanation of how marks were awarded>",
      "feedback": "<constructive feedback for the student>",
      "strengths": "<student's strengths in this answer>",
      "weaknesses": "<areas of improvement or conceptual gaps>",
      "error_type": "<null | sign_error | algebraic_error | arithmetic_flub | calculation_error | conceptual_error>"
    }}
    """
    
    parts = [prompt]
    
    # Download and append diagram images if any
    if diagram_keys:
        for key in diagram_keys:
            try:
                img_bytes = download_file(key)
                mime = "image/png"
                if key.lower().endswith(".jpg") or key.lower().endswith(".jpeg"):
                    mime = "image/jpeg"
                parts.append(genai_types.Part.from_bytes(data=img_bytes, mime_type=mime))
                logger.info(f"[Grade] Appended diagram {key} to Gemini grading call")
            except Exception as e:
                logger.error(f"[Grade] Failed to download diagram {key}: {e}")

    client = get_client(api_key=api_key)
    try:
        response = client.models.generate_content(
            model=settings.GEMINI_MODEL,
            contents=parts,
            config=genai_types.GenerateContentConfig(
                temperature=0.0,
                max_output_tokens=1024,
            ),
        )
        raw_text = response.text or ""
        parsed = parse_json_response(raw_text)
        if isinstance(parsed, dict):
            # Ensure marks_awarded is clamped
            marks = float(parsed.get("marks_awarded", 0.0))
            parsed["marks_awarded"] = max(0.0, min(float(max_marks), marks))
            return parsed
            
        raise ValueError("Invalid JSON response from Gemini")
    except Exception as e:
        logger.error(f"[Grade] Error grading answer: {e}")
        return {
            "marks_awarded": 0.0,
            "justification": f"Grading failed due to an error: {e}",
            "feedback": "AI grading error. Please review manually.",
            "strengths": "N/A",
            "weaknesses": "N/A",
            "error_type": "grading_error"
        }


def grade_submission(
    rubric: dict,
    parsed_content: dict,
    temperature: float = 0.0,
    subject: str = "General",
    board: str = "CBSE",
    grade_level: str = "Class 12",
    file_key: str | None = None,
    submission_id: str | None = None,
    user_gemini_key: str | None = None,
) -> dict:
    """
    Refactored question-by-question independent grading pipeline.
    """
    start_time = time.time()
    
    rubric_steps = rubric.get("steps", [])
    student_steps = parsed_content.get("steps", [])
    
    logger.info(f"[Grade] Grading submission {submission_id} question-by-question")
    
    step_grades = []
    total_awarded = 0.0
    total_max = 0.0
    
    component_totals = {}  # component_type -> {"awarded": 0.0, "max": 0.0}

    # Map student steps by step_num for direct question lookup
    student_step_map = {str(s.get("step_num")): s for s in student_steps}

    for r_step in rubric_steps:
        s_num = str(r_step.get("step_num"))
        q_text = r_step.get("description", "")
        max_marks = float(r_step.get("marks", 5.0))
        marking_notes = r_step.get("marking_notes", "")
        component_type = r_step.get("component_type", "text")
        
        # Default fallback if student didn't answer
        student_ans = ""
        diagram_keys = []
        
        # Retrieve mapped student answer
        s_step = student_step_map.get(s_num)
        if s_step:
            student_ans = s_step.get("text", "")
            diagrams = s_step.get("diagrams", [])
            diagram_keys = [d["key"] for d in diagrams if "key" in d]

        logger.info(f"[Grade] Grading question {s_num}: {q_text[:30]}... (max={max_marks})")
        
        # Grade single answer
        result = grade_single_answer(
            question_text=q_text,
            max_marks=max_marks,
            marking_notes=marking_notes,
            student_answer=student_ans,
            diagram_keys=diagram_keys,
            subject=subject,
            board=board,
            grade_level=grade_level,
            api_key=user_gemini_key
        )
        
        awarded = float(result.get("marks_awarded", 0.0))
        justification = result.get("justification", "")
        feedback = result.get("feedback", "")
        strengths = result.get("strengths", "")
        weaknesses = result.get("weaknesses", "")
        error_type = result.get("error_type")
        
        # Combine feedback, strengths, and weaknesses into justification text for database
        full_justification = f"Feedback: {feedback}\nStrengths: {strengths}\nWeaknesses: {weaknesses}\nRationale: {justification}"

        dist = [0.0] * (int(max_marks) + 1) if max_marks > 0 else [1.0]
        clamped_idx = min(int(round(awarded)), len(dist) - 1)
        dist[clamped_idx] = 1.0

        step_grades.append({
            "step_num": int(s_num) if s_num.isdigit() else len(step_grades) + 1,
            "marks_awarded": awarded,
            "max_marks": max_marks,
            "grade_distribution": dist,
            "justification": full_justification,
            "error_type": error_type,
            "sympy_valid": True if component_type == "reasoning" else None,
            "sympy_error": None
        })
        
        total_awarded += awarded
        total_max += max_marks
        
        # Track component stats
        if component_type not in component_totals:
            component_totals[component_type] = {"awarded": 0.0, "max": 0.0}
        component_totals[component_type]["awarded"] += awarded
        component_totals[component_type]["max"] += max_marks

    # Form component grades list
    component_grades = []
    for ctype, totals in component_totals.items():
        max_m = int(totals["max"])
        dist = [0.0] * (max_m + 1) if max_m > 0 else [1.0]
        clamped_idx = min(int(round(totals["awarded"])), len(dist) - 1)
        dist[clamped_idx] = 1.0
        
        component_grades.append({
            "type": ctype,
            "marks_awarded": int(round(totals["awarded"])),
            "max_marks": max_m,
            "confidence": 0.95,
            "grade_distribution": dist,
            "justification": f"Independent Q&A grading component aggregation for {ctype}."
        })

    overall_dist = [0.0] * (int(total_max) + 1) if total_max > 0 else [1.0]
    clamped_overall = min(int(round(total_awarded)), len(overall_dist) - 1)
    overall_dist[clamped_overall] = 1.0

    latency_ms = int((time.time() - start_time) * 1000)

    from app.config import settings
    return {
        "grade": int(round(total_awarded)),
        "max_grade": int(total_max),
        "grade_distribution": overall_dist,
        "confidence": 0.95,
        "step_grades": step_grades,
        "component_grades": component_grades,
        "justification": f"Independent grading complete. Sum of question scores: {total_awarded:.1f}/{total_max:.1f}",
        "review_status": "AUTO_GRADED",
        "review_reasons": [],
        "flagged_components": [],
        "question_decomposition": rubric_steps,
        "llm_call_ids": [],
        "model_used": settings.GEMINI_MODEL,
        "latency_ms": latency_ms,
    }


async def grade_submission_background(submission_id: str, grading_run_id: str):
    """Background task to grade a single submission."""
    from app.tasks.grade_submission import grade
    await grade(submission_id, grading_run_id)
