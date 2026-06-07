"""
LLM Client — Multi-provider AI wrapper.

Provider strategy (LLM_PROVIDER_STRATEGY):
  "openrouter_primary" → OpenRouter first, Gemini fallback       (default)
  "claude_primary"     → Claude first, Gemini fallback
  "gemini_primary"     → Gemini first, Claude fallback
  "openrouter_only"    → OpenRouter only, no fallback
  "gemini_only"        → Gemini only
  "claude_only"        → Claude only
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

_gemini_client: genai.Client | None = None
_gemini_client_key: str | None = None


def get_client(api_key: str | None = None) -> genai.Client:
    """Get or create a Gemini client. Supports BYOK via api_key override."""
    global _gemini_client, _gemini_client_key
    key = api_key or settings.GEMINI_API_KEY
    if api_key:
        return genai.Client(api_key=api_key)
    if _gemini_client is None or _gemini_client_key != key:
        _gemini_client = genai.Client(api_key=key) if key else genai.Client()
        _gemini_client_key = key
    return _gemini_client


# ─── OpenRouter provider ─────────────────────────────────────────────────────

def _call_openrouter_raw(
    prompt: str,
    system_prompt: str = "",
    temperature: float = 0.0,
    max_tokens: int = 2048,
    image_bytes: bytes | list[bytes] | None = None,
    image_mime: str | list[str] = "image/png",
    api_key: str | None = None,
) -> dict[str, Any]:
    """
    Single call to OpenRouter (OpenAI-compatible REST API).
    Supports vision via base64-encoded image in the messages array.
    """
    import httpx

    key = api_key or settings.OPENROUTER_API_KEY
    if not key:
        return {
            "response_text": "", "tokens_in": 0, "tokens_out": 0,
            "latency_ms": 0, "model": settings.OPENROUTER_MODEL,
            "success": False, "error": "OPENROUTER_API_KEY not configured",
        }

    model = settings.OPENROUTER_MODEL
    messages: list[dict] = []

    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    # Build user message — text only or multimodal
    if image_bytes:
        image_bytes_list = [image_bytes] if isinstance(image_bytes, bytes) else image_bytes
        if isinstance(image_mime, str):
            image_mime_list = [image_mime] * len(image_bytes_list)
        else:
            image_mime_list = image_mime

        content_parts = []
        for img_b, img_m in zip(image_bytes_list, image_mime_list):
            b64 = base64.standard_b64encode(img_b).decode("utf-8")
            content_parts.append({"type": "image_url", "image_url": {"url": f"data:{img_m};base64,{b64}"}})
        content_parts.append({"type": "text", "text": prompt})

        messages.append({
            "role": "user",
            "content": content_parts,
        })
    else:
        messages.append({"role": "user", "content": prompt})

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://ozymorlab.com",
        "X-Title": "OzymorLab",
    }

    try:
        start = time.time()
        response = httpx.post(
            "https://openrouter.ai/api/v1/chat/completions",
            json=payload,
            headers=headers,
            timeout=120.0,
        )
        latency_ms = int((time.time() - start) * 1000)

        if response.status_code != 200:
            return {
                "response_text": "", "tokens_in": 0, "tokens_out": 0,
                "latency_ms": latency_ms, "model": model,
                "success": False, "error": f"HTTP {response.status_code}: {response.text[:200]}",
            }

        data = response.json()
        choice = data.get("choices", [{}])[0]
        response_text = choice.get("message", {}).get("content", "") or ""
        usage = data.get("usage", {})
        tokens_in = usage.get("prompt_tokens", 0)
        tokens_out = usage.get("completion_tokens", 0)

        logger.info(
            f"OpenRouter [{model}] {latency_ms}ms "
            f"({tokens_in}/{tokens_out} tokens)"
        )
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
            "latency_ms": 0, "model": model,
            "success": False, "error": str(e),
        }


# ─── Claude provider ─────────────────────────────────────────────────────────

def _get_anthropic_client(api_key: str | None = None):
    try:
        import anthropic  # type: ignore
    except ImportError:
        raise RuntimeError("anthropic package is not installed. Run: pip install anthropic")
    key = api_key or settings.CLAUDE_API_KEY
    if not key:
        raise RuntimeError("CLAUDE_API_KEY is not configured.")
    return anthropic.Anthropic(api_key=key)


def _call_claude_raw(
    prompt: str,
    system_prompt: str = "",
    temperature: float = 0.0,
    max_tokens: int = 2048,
    image_bytes: bytes | list[bytes] | None = None,
    image_mime: str | list[str] = "image/png",
    api_key: str | None = None,
) -> dict[str, Any]:
    client = _get_anthropic_client(api_key)
    model = settings.CLAUDE_MODEL
    content: list = []
    if image_bytes:
        image_bytes_list = [image_bytes] if isinstance(image_bytes, bytes) else image_bytes
        if isinstance(image_mime, str):
            image_mime_list = [image_mime] * len(image_bytes_list)
        else:
            image_mime_list = image_mime

        for img_b, img_m in zip(image_bytes_list, image_mime_list):
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": img_m,
                    "data": base64.standard_b64encode(img_b).decode("utf-8"),
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
            "response_text": response_text, "tokens_in": tokens_in,
            "tokens_out": tokens_out, "latency_ms": latency_ms,
            "model": model, "success": True,
        }
    except Exception as e:
        return {
            "response_text": "", "tokens_in": 0, "tokens_out": 0,
            "latency_ms": 0, "model": model, "success": False, "error": str(e),
        }


# ─── Gemini provider ─────────────────────────────────────────────────────────

def _call_gemini_raw(
    prompt: str,
    system_prompt: str = "",
    temperature: float = 0.0,
    max_tokens: int = 2048,
    image_bytes: bytes | list[bytes] | None = None,
    image_mime: str | list[str] = "image/png",
    call_type: str = "general",
    api_key: str | None = None,
) -> dict[str, Any]:
    client = get_client(api_key=api_key)
    model = settings.GEMINI_MODEL
    contents: list = []
    if image_bytes:
        image_bytes_list = [image_bytes] if isinstance(image_bytes, bytes) else image_bytes
        if isinstance(image_mime, str):
            image_mime_list = [image_mime] * len(image_bytes_list)
        else:
            image_mime_list = image_mime

        for img_b, img_m in zip(image_bytes_list, image_mime_list):
            contents.append(types.Part.from_bytes(data=img_b, mime_type=img_m))
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
            "response_text": response_text, "tokens_in": tokens_in,
            "tokens_out": tokens_out, "latency_ms": latency_ms,
            "model": model, "success": True,
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
    image_bytes: bytes | list[bytes] | None = None,
    image_mime: str | list[str] = "image/png",
    api_key: str | None = None,
) -> dict[str, Any]:
    """
    Unified LLM call respecting LLM_PROVIDER_STRATEGY.

    Strategies:
      openrouter_primary → OpenRouter → Gemini fallback  (default)
      claude_primary     → Claude    → Gemini fallback
      gemini_primary     → Gemini    → Claude fallback
      openrouter_only    → OpenRouter only
      claude_only        → Claude only
      gemini_only        → Gemini only (with retries)
    """
    temp = temperature if temperature is not None else settings.GRADING_TEMPERATURE
    strategy = settings.LLM_PROVIDER_STRATEGY.lower()

    def _openrouter() -> dict[str, Any]:
        return _call_openrouter_raw(
            prompt, system_prompt, temp, max_tokens, image_bytes, image_mime, api_key
        )

    def _claude() -> dict[str, Any]:
        try:
            return _call_claude_raw(
                prompt, system_prompt, temp, max_tokens, image_bytes, image_mime, api_key
            )
        except RuntimeError as e:
            return {
                "response_text": "", "tokens_in": 0, "tokens_out": 0,
                "latency_ms": 0, "model": settings.CLAUDE_MODEL,
                "success": False, "error": str(e),
            }

    def _gemini() -> dict[str, Any]:
        return _call_gemini_raw(
            prompt, system_prompt, temp, max_tokens, image_bytes, image_mime, call_type, api_key
        )

    # Single-provider strategies
    if strategy == "openrouter_only":
        return _openrouter()
    if strategy == "claude_only":
        return _claude()
    if strategy == "gemini_only":
        return call_gemini(prompt, system_prompt, temp, max_tokens, call_type, api_key)

    # Two-provider strategies with fallback
    strategy_map = {
        "openrouter_primary": (_openrouter, _gemini,    "OpenRouter", "Gemini"),
        "claude_primary":     (_claude,     _gemini,    "Claude",     "Gemini"),
        "gemini_primary":     (_gemini,     _claude,    "Gemini",     "Claude"),
    }
    primary_fn, fallback_fn, primary_name, fallback_name = strategy_map.get(
        strategy,
        (_openrouter, _gemini, "OpenRouter", "Gemini"),  # default
    )

    result = primary_fn()
    if result["success"]:
        return result

    logger.warning(
        f"[{call_type}] {primary_name} failed ({result.get('error', 'unknown')}). "
        f"Falling back to {fallback_name}."
    )
    fallback_result = fallback_fn()
    if not fallback_result["success"]:
        logger.error(
            f"[{call_type}] {fallback_name} fallback also failed: "
            f"{fallback_result.get('error', 'unknown')}"
        )
    return fallback_result


# ─── Legacy call_gemini (kept for backward compat) ───────────────────────────

def call_gemini(
    prompt: str,
    system_prompt: str = "",
    temperature: float | None = None,
    max_tokens: int = 2048,
    call_type: str = "general",
    api_key: str | None = None,
) -> dict[str, Any]:
    """Legacy Gemini call with retry. Prefer call_llm() for new code."""
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
            logger.warning(f"Gemini [{call_type}] attempt {attempt + 1} failed: {e}. Retry in {wait_time:.1f}s")
            time.sleep(wait_time)
    logger.error(f"Gemini [{call_type}] failed after {settings.GRADING_MAX_RETRIES} attempts")
    return {
        "response_text": "", "tokens_in": 0, "tokens_out": 0,
        "latency_ms": 0, "model": model, "success": False, "error": str(last_error),
    }


# ─── Utilities ────────────────────────────────────────────────────────────────

def get_grading_system_prompt(
    subject: str = "General",
    board: str = "Generic",
    grade_level: str = "Unknown",
) -> str:
    """Load the base system prompt and append all markdown KB files."""
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
    """Use vision-capable LLM to transcribe handwritten text and math."""
    result = call_llm(
        prompt=(
            "Extract all the handwritten text, mathematical equations, and derivations "
            "from this image exactly as written. Do not summarize. Just output the transcribed text."
        ),
        image_bytes=image_bytes,
        image_mime="image/jpeg",
        call_type="ocr_image",
    )
    if result["success"]:
        return result["response_text"]
    logger.error(f"Vision OCR failed: {result.get('error')}")
    raise RuntimeError(f"Vision OCR failed: {result.get('error')}")


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
