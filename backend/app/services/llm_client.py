"""
LLM Client — Multi-provider AI wrapper with Claude primary, Gemini fallback.

Provider strategy is controlled by LLM_PROVIDER_STRATEGY in settings:
  "claude_primary"  → Claude first, Gemini fallback  (default)
  "gemini_primary"  → Gemini first, Claude fallback
  "gemini_only"     → Gemini only, no fallback
  "claude_only"     → Claude only, no fallback
"""
import json
import time
import base64
import logging
import os
from typing import Any

from google import genai
from google.genai import types

from app.config import settings

logger = logging.getLogger(__name__)

# ─── Gemini client cache ─────────────────────────────────────────────────────

_client: genai.Client | None = None
_client_key: str | None = None


def get_client(api_key: str | None = None) -> genai.Client:
    """Get or create a Gemini client. Supports BYOK via api_key override."""
    global _client, _client_key
    key = api_key or settings.GEMINI_API_KEY

    # If a BYOK key is provided, always create a fresh client (don't pollute cache)
    if api_key:
        return genai.Client(api_key=api_key)

    # Use cached system client
    if _client is None or _client_key != key:
        if key:
            _client = genai.Client(api_key=key)
        else:
            _client = genai.Client()  # Fall back to SDK's automatic env var checking
        _client_key = key
    return _client


# ─── Claude helpers ──────────────────────────────────────────────────────────

def _get_anthropic_client(api_key: str | None = None):
    """Lazy-import anthropic and return a client."""
    try:
        import anthropic  # type: ignore
    except ImportError:
        raise RuntimeError(
            "anthropic package is not installed. Run: pip install anthropic"
        )
    key = api_key or settings.CLAUDE_API_KEY
    if not key:
        raise RuntimeError("CLAUDE_API_KEY is not configured.")
    return anthropic.Anthropic(api_key=key)


def _call_claude_raw(
    prompt: str,
    system_prompt: str = "",
    temperature: float = 0.0,
    max_tokens: int = 2048,
    image_bytes: bytes | None = None,
    image_mime: str = "image/png",
    api_key: str | None = None,
) -> dict[str, Any]:
    """Single call to Claude. Returns same shape as call_gemini()."""
    client = _get_anthropic_client(api_key)
    model = settings.CLAUDE_MODEL

    # Build content list
    content: list = []
    if image_bytes:
        content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": image_mime,
                "data": base64.standard_b64encode(image_bytes).decode("utf-8"),
            },
        })
    content.append({"type": "text", "text": prompt})

    try:
        import anthropic  # type: ignore
        start = time.time()
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system_prompt if system_prompt else anthropic.NOT_GIVEN,
            messages=[{"role": "user", "content": content}],
            temperature=temperature,
        )
        latency_ms = int((time.time() - start) * 1000)
        response_text = response.content[0].text if response.content else ""
        tokens_in = response.usage.input_tokens if response.usage else 0
        tokens_out = response.usage.output_tokens if response.usage else 0
        logger.info(f"Claude {latency_ms}ms ({tokens_in}/{tokens_out} tokens)")
        return {
            "response_text": response_text,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "latency_ms": latency_ms,
            "model": model,
            "success": True,
        }
    except Exception as e:
        return {
            "response_text": "", "tokens_in": 0, "tokens_out": 0,
            "latency_ms": 0, "model": model, "success": False, "error": str(e),
        }


def _call_gemini_raw(
    prompt: str,
    system_prompt: str = "",
    temperature: float = 0.0,
    max_tokens: int = 2048,
    image_bytes: bytes | None = None,
    image_mime: str = "image/png",
    call_type: str = "general",
    api_key: str | None = None,
) -> dict[str, Any]:
    """Single-attempt Gemini call (no retry). Returns same shape as call_gemini()."""
    client = get_client(api_key=api_key)
    model = settings.GEMINI_MODEL

    contents: list = []
    if image_bytes:
        contents.append(types.Part.from_bytes(data=image_bytes, mime_type=image_mime))
    contents.append(prompt)

    try:
        start = time.time()
        response = client.models.generate_content(
            model=model,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt or None,
                temperature=temperature,
                max_output_tokens=max_tokens,
            ),
        )
        latency_ms = int((time.time() - start) * 1000)
        response_text = response.text or ""
        tokens_in = tokens_out = 0
        if response.usage_metadata:
            tokens_in = response.usage_metadata.prompt_token_count or 0
            tokens_out = response.usage_metadata.candidates_token_count or 0
        logger.info(f"Gemini [{call_type}] {latency_ms}ms ({tokens_in}/{tokens_out} tokens)")
        return {
            "response_text": response_text,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "latency_ms": latency_ms,
            "model": model,
            "success": True,
        }
    except Exception as e:
        return {
            "response_text": "", "tokens_in": 0, "tokens_out": 0,
            "latency_ms": 0, "model": model, "success": False, "error": str(e),
        }


# ─── Unified call_llm ────────────────────────────────────────────────────────

def call_llm(
    prompt: str,
    system_prompt: str = "",
    temperature: float | None = None,
    max_tokens: int = 2048,
    call_type: str = "general",
    image_bytes: bytes | None = None,
    image_mime: str = "image/png",
    api_key: str | None = None,
) -> dict[str, Any]:
    """
    Unified LLM call respecting LLM_PROVIDER_STRATEGY.

    Strategy routing:
      claude_primary  → Claude → Gemini fallback  (default)
      gemini_primary  → Gemini → Claude fallback
      claude_only     → Claude only
      gemini_only     → Gemini only (with retries via call_gemini)

    image_bytes/image_mime are forwarded for vision tasks (OCR, diagram detection).
    """
    temp = temperature if temperature is not None else settings.GRADING_TEMPERATURE
    strategy = settings.LLM_PROVIDER_STRATEGY.lower()

    def _claude() -> dict[str, Any]:
        try:
            return _call_claude_raw(prompt, system_prompt, temp, max_tokens,
                                    image_bytes, image_mime, api_key)
        except RuntimeError as e:
            # Key not configured or package missing — treat as provider failure
            return {"response_text": "", "tokens_in": 0, "tokens_out": 0,
                    "latency_ms": 0, "model": settings.CLAUDE_MODEL,
                    "success": False, "error": str(e)}

    def _gemini() -> dict[str, Any]:
        return _call_gemini_raw(prompt, system_prompt, temp, max_tokens,
                                image_bytes, image_mime, call_type, api_key)

    if strategy == "claude_only":
        return _claude()

    if strategy == "gemini_only":
        # Keep original retry behaviour for gemini-only mode
        return call_gemini(prompt, system_prompt, temp, max_tokens, call_type, api_key)

    if strategy == "gemini_primary":
        primary, fallback, primary_name, fallback_name = _gemini, _claude, "Gemini", "Claude"
    else:
        # Default: claude_primary
        primary, fallback, primary_name, fallback_name = _claude, _gemini, "Claude", "Gemini"

    result = primary()
    if result["success"]:
        return result

    logger.warning(
        f"[{call_type}] {primary_name} failed ({result.get('error', 'unknown')}). "
        f"Falling back to {fallback_name}."
    )
    fallback_result = fallback()
    if not fallback_result["success"]:
        logger.error(
            f"[{call_type}] {fallback_name} fallback also failed: "
            f"{fallback_result.get('error', 'unknown')}"
        )
    return fallback_result


# ─── Legacy call_gemini (kept for backward compat & gemini_only strategy) ────

def get_grading_system_prompt(
    subject: str = "General",
    board: str = "Generic",
    grade_level: str = "Unknown",
) -> str:
    """Load the base system prompt dynamically and append all markdown KB files."""
    base_prompt = (
        f"You are an expert {board} examiner ({grade_level} {subject}) with 15 years of experience.\n"
        f"You grade subjective answers and derivations step by step, awarding partial marks according to the {board} marking scheme.\n"
        "You output ONLY valid JSON — no preamble, no markdown, no code fences.\n\n"
        "Grading philosophy:\n"
        "- Award marks for correct method even if the final answer is wrong\n"
        "- Penalize errors in intermediate steps that cascade\n"
        "- Accept equivalent mathematical expressions (e.g., F=ma and a=F/m are equivalent)\n"
        "- If a step is partially correct, award proportional partial credit"
    )

    prompt = base_prompt
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


ALIGNMENT_SYSTEM_PROMPT = (
    "You are an expert at analyzing student answers and mapping them to rubric steps.\n"
    "Given rubric steps and student answer steps, map each rubric step to the most relevant student step(s).\n"
    "Output ONLY valid JSON — no preamble, no markdown, no code fences."
)


def extract_text_from_image_gemini(image_bytes: bytes) -> str:
    """Use Gemini Vision to transcribe fuzzy handwritten text and math."""
    client = get_client()
    try:
        response = client.models.generate_content(
            model=settings.GEMINI_MODEL,
            contents=[
                types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
                "Extract all the handwritten text, mathematical equations, and derivations "
                "from this image exactly as written. Do not summarize. Just output the transcribed text.",
            ],
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
    api_key: str | None = None,
) -> dict[str, Any]:
    """Make a call to Gemini API with retry logic and exponential backoff.

    Args:
        api_key: Optional BYOK Gemini API key. If provided, uses this key
                 instead of the system-wide key from settings.
    """
    client = get_client(api_key=api_key)
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
            logger.warning(
                f"Gemini [{call_type}] attempt {attempt + 1} failed: {e}. "
                f"Retry in {wait_time:.1f}s"
            )
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


def build_step_grading_prompt(
    rubric_step: dict,
    student_step: dict,
    sympy_result: dict | None = None,
    board_notes: str = "",
) -> str:
    """Build the grading prompt for a single rubric step."""
    sympy_ctx = ""
    if sympy_result:
        if sympy_result.get("valid") is True:
            sympy_ctx = "[SYMBOLIC VALIDATION: Equation transformation is mathematically correct]"
        elif sympy_result.get("valid") is False:
            sympy_ctx = f"[SYMBOLIC VALIDATION: ERROR — {sympy_result.get('error', 'Unknown')}]"

    mm = rubric_step.get("marks", 1)
    return (
        f"RUBRIC STEP {rubric_step.get('step_num', '?')} (max {mm} marks):\n"
        f"Description: {rubric_step.get('description', '')}\n"
        f"Marking notes: {rubric_step.get('marking_notes', '')}\n"
        f"Partial credit: {rubric_step.get('partial_credit', True)}\n"
        f"{f'Board guidance: {board_notes}' if board_notes else ''}\n\n"
        f"STUDENT ANSWER FOR THIS STEP:\n"
        f"{student_step.get('text', '')}\n\n"
        f"{sympy_ctx}\n\n"
        f"Grade this step. Return JSON:\n"
        f'{{"marks_awarded": <int 0..{mm}>, "grade_distribution": <array of {mm + 1} floats summing to 1.0>, '
        f'"justification": "<one sentence>", "error_type": "<null|algebraic_error|missing_step|wrong_formula|presentation>"}}'
    )


def build_alignment_prompt(rubric_steps: list[dict], student_steps: list[dict]) -> str:
    """Build prompt to align student steps to rubric steps."""
    r_desc = "\n".join(
        f"  Rubric Step {s.get('step_num', i + 1)}: {s.get('description', '')}"
        for i, s in enumerate(rubric_steps)
    )
    s_desc = "\n".join(
        f"  Student Step {s.get('step_num', i + 1)}: {s.get('text', '')[:200]}"
        for i, s in enumerate(student_steps)
    )
    return (
        f"RUBRIC STEPS:\n{r_desc}\n\nSTUDENT STEPS:\n{s_desc}\n\n"
        "Map each rubric step to the most relevant student step(s). "
        "Use only the integer step numbers for the keys and values. Return a JSON array:\n"
        "[\n"
        '  {"rubric_step": 1, "student_steps": [2, 3], "confidence": 0.95}\n'
        "]\n"
        'If no match, use "student_steps": [] with confidence 0.0.'
    )
