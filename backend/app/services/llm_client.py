"""
LLM Client — Google Gemini API wrapper with retry logic.
Uses Gemini 2.5 Pro for grading physics derivations.
"""
import json
import time
import logging
import os
from typing import Any

from google import genai
from google.genai import types

from app.config import settings

logger = logging.getLogger(__name__)

_client: genai.Client | None = None


def get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(api_key=settings.GEMINI_API_KEY)
    return _client


def get_grading_system_prompt(subject: str = "General", board: str = "Generic", grade_level: str = "Unknown") -> str:
    """Load the base system prompt dynamically based on the subject and append all markdown KB files."""
    
    base_prompt = f"""You are an expert {board} examiner ({grade_level} {subject}) with 15 years of experience.
You grade subjective answers and derivations step by step, awarding partial marks according to the {board} marking scheme.
You output ONLY valid JSON — no preamble, no markdown, no code fences.

Grading philosophy:
- Award marks for correct method even if the final answer is wrong
- Penalize errors in intermediate steps that cascade
- Accept equivalent mathematical expressions (e.g., F=ma and a=F/m are equivalent)
- If a step is partially correct, award proportional partial credit"""

    prompt = base_prompt
    
    # Path to the kb directory (backend/kb)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    kb_dir = os.path.join(os.path.dirname(os.path.dirname(current_dir)), "kb")
    
    if os.path.exists(kb_dir):
        prompt += "\n\n--- ADDITIONAL GRADING GUIDELINES (KNOWLEDGE BASE) ---"
        for filename in sorted(os.listdir(kb_dir)):
            if filename.endswith(".md"):
                filepath = os.path.join(kb_dir, filename)
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        prompt += f"\n\n### Document: {filename}\n"
                        prompt += f.read()
                except Exception as e:
                    logger.error(f"Failed to load KB file {filename}: {e}")
                    
    return prompt

ALIGNMENT_SYSTEM_PROMPT = """You are an expert at analyzing student answers and mapping them to rubric steps.
Given rubric steps and student answer steps, map each rubric step to the most relevant student step(s).
Output ONLY valid JSON — no preamble, no markdown, no code fences."""

def extract_text_from_image_gemini(image_bytes: bytes) -> str:
    """Use Gemini 2.5 Pro Vision to transcribe fuzzy handwritten text and math."""
    client = get_client()
    try:
        response = client.models.generate_content(
            model=settings.GEMINI_MODEL,
            contents=[
                types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
                "Extract all the handwritten text, mathematical equations, and derivations from this image exactly as written. Do not summarize. Just output the transcribed text."
            ]
        )
        return response.text or ""
    except Exception as e:
        logger.error(f"Gemini Vision OCR failed: {e}")
        raise



def call_gemini(
    prompt: str,
    system_prompt: str = "",
    temperature: float | None = None,
    max_tokens: int = 2048,
    call_type: str = "general",
) -> dict[str, Any]:
    """Make a call to Gemini API with retry logic and exponential backoff."""
    client = get_client()
    temp = temperature if temperature is not None else settings.GRADING_TEMPERATURE
    model = settings.GEMINI_MODEL

    last_error = None
    for attempt in range(settings.GRADING_MAX_RETRIES):
        try:
            start_time = time.time()
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt if system_prompt else None,
                    temperature=temp,
                    max_output_tokens=max_tokens,
                ),
            )
            latency_ms = int((time.time() - start_time) * 1000)
            response_text = response.text or ""

            tokens_in = tokens_out = 0
            if response.usage_metadata:
                tokens_in = response.usage_metadata.prompt_token_count or 0
                tokens_out = response.usage_metadata.candidates_token_count or 0

            logger.info(f"Gemini [{call_type}] {latency_ms}ms ({tokens_in}/{tokens_out} tokens)")
            return {
                "response_text": response_text, "tokens_in": tokens_in,
                "tokens_out": tokens_out, "latency_ms": latency_ms,
                "model": model, "success": True,
            }
        except Exception as e:
            last_error = e
            wait_time = (2 ** attempt) + 0.5
            logger.warning(f"Gemini [{call_type}] attempt {attempt+1} failed: {e}. Retry in {wait_time:.1f}s")
            time.sleep(wait_time)

    logger.error(f"Gemini [{call_type}] failed after {settings.GRADING_MAX_RETRIES} attempts")
    return {
        "response_text": "", "tokens_in": 0, "tokens_out": 0,
        "latency_ms": 0, "model": model, "success": False, "error": str(last_error),
    }


def parse_json_response(response_text: str) -> dict | list | None:
    """Parse JSON from LLM response, handling markdown fences."""
    if not response_text:
        return None
    text = response_text.strip()
    if text.startswith("```"):
        first_nl = text.index("\n") if "\n" in text else len(text)
        text = text[first_nl + 1:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        for sc, ec in [("{", "}"), ("[", "]")]:
            si, ei = text.find(sc), text.rfind(ec)
            if si != -1 and ei > si:
                try:
                    return json.loads(text[si:ei + 1])
                except json.JSONDecodeError:
                    continue
        logger.warning(f"JSON parse failed: {text[:200]}...")
        return None


def build_step_grading_prompt(rubric_step: dict, student_step: dict,
                               sympy_result: dict | None = None, board_notes: str = "") -> str:
    """Build the grading prompt for a single rubric step."""
    sympy_ctx = ""
    if sympy_result:
        if sympy_result.get("valid") is True:
            sympy_ctx = "[SYMBOLIC VALIDATION: Equation transformation is mathematically correct]"
        elif sympy_result.get("valid") is False:
            sympy_ctx = f"[SYMBOLIC VALIDATION: ERROR — {sympy_result.get('error', 'Unknown')}]"

    mm = rubric_step.get("marks", 1)
    return f"""RUBRIC STEP {rubric_step.get('step_num', '?')} (max {mm} marks):
Description: {rubric_step.get('description', '')}
Marking notes: {rubric_step.get('marking_notes', '')}
Partial credit: {rubric_step.get('partial_credit', True)}
{f'Board guidance: {board_notes}' if board_notes else ''}

STUDENT ANSWER FOR THIS STEP:
{student_step.get('text', '')}

{sympy_ctx}

Grade this step. Return JSON:
{{"marks_awarded": <int 0..{mm}>, "grade_distribution": <array of {mm+1} floats summing to 1.0>, "justification": "<one sentence>", "error_type": "<null|algebraic_error|missing_step|wrong_formula|presentation>"}}"""


def build_alignment_prompt(rubric_steps: list[dict], student_steps: list[dict]) -> str:
    """Build prompt to align student steps to rubric steps."""
    r_desc = "\n".join(f"  R{s.get('step_num', i+1)}: {s.get('description', '')}" for i, s in enumerate(rubric_steps))
    s_desc = "\n".join(f"  S{s.get('step_num', i+1)}: {s.get('text', '')[:200]}" for i, s in enumerate(student_steps))
    return f"""RUBRIC STEPS:\n{r_desc}\n\nSTUDENT STEPS:\n{s_desc}\n
Map each rubric step to relevant student step(s). Return JSON array:
[{{"rubric_step": 1, "student_steps": [1, 2], "confidence": 0.9}}, ...]
If no match, use "student_steps": [] with confidence 0.0."""
