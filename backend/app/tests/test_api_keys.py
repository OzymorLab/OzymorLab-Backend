"""
API Key Connectivity Tests — verifies that configured LLM provider keys
are valid and can reach their respective APIs.

Run with:
    pytest backend/app/tests/test_api_keys.py -v

These are LIVE tests — they make real network calls.
They are skipped automatically when keys are not configured.
"""
import pytest
from app.config import settings


# ─── Gemini ──────────────────────────────────────────────────────────────────

@pytest.mark.skipif(
    not settings.GEMINI_API_KEY,
    reason="GEMINI_API_KEY is not configured — skipping",
)
def test_gemini_api_key_is_valid():
    """
    Sends a minimal prompt to Gemini and asserts a non-empty response.
    A 429 (quota exhausted) is treated as a key-is-valid result —
    the key exists but the account is out of credits.
    """
    from app.services.llm_client import _call_gemini_raw

    result = _call_gemini_raw(
        prompt="Reply with the single word: PONG",
        temperature=0.0,
        max_tokens=16,
        call_type="key_test",
    )

    # 429 quota error → key is valid, account is depleted
    error_str = str(result.get("error", "")).lower()
    is_quota_error = (
        "429" in error_str
        or "resource_exhausted" in error_str
        or "quota" in error_str
        or "prepayment" in error_str
    )

    if is_quota_error:
        pytest.skip(
            f"Gemini key is valid but account quota is exhausted: {result.get('error')}"
        )

    assert result["success"], (
        f"Gemini API key check failed.\n"
        f"Error: {result.get('error')}\n"
        f"Tip: Check GEMINI_API_KEY in your .env"
    )
    assert result["response_text"].strip(), "Gemini returned an empty response"
    print(f"\n✅ Gemini OK | model={result['model']} | "
          f"{result['latency_ms']}ms | response='{result['response_text'].strip()}'")


# ─── Claude ──────────────────────────────────────────────────────────────────

@pytest.mark.skipif(
    not settings.CLAUDE_API_KEY,
    reason="CLAUDE_API_KEY is not configured — skipping",
)
def test_claude_api_key_is_valid():
    """
    Sends a minimal prompt to Claude and asserts a non-empty response.
    Distinguishes between an invalid key (401) and a quota/billing issue (429).
    """
    from app.services.llm_client import _call_claude_raw

    result = _call_claude_raw(
        prompt="Reply with the single word: PONG",
        temperature=0.0,
        max_tokens=16,
    )

    error_str = str(result.get("error", "")).lower()

    # Billing / rate limit → key is valid but account needs attention
    is_billing_error = (
        "credit" in error_str
        or "billing" in error_str
        or "429" in error_str
        or "overloaded" in error_str
        or "rate_limit" in error_str
    )
    # Authentication failure → key is wrong
    is_auth_error = (
        "401" in error_str
        or "authentication" in error_str
        or "invalid x-api-key" in error_str
        or "permission" in error_str
    )

    if is_billing_error:
        pytest.skip(
            f"Claude key is valid but account has billing/rate issues: {result.get('error')}"
        )

    assert not is_auth_error, (
        f"Claude API key is invalid or unauthorized.\n"
        f"Error: {result.get('error')}\n"
        f"Tip: Check CLAUDE_API_KEY in your .env"
    )
    assert result["success"], (
        f"Claude API key check failed.\n"
        f"Error: {result.get('error')}\n"
        f"Tip: Check CLAUDE_API_KEY in your .env"
    )
    assert result["response_text"].strip(), "Claude returned an empty response"
    print(f"\n✅ Claude OK | model={result['model']} | "
          f"{result['latency_ms']}ms | response='{result['response_text'].strip()}'")


# ─── Strategy / Fallback ─────────────────────────────────────────────────────

@pytest.mark.skipif(
    not settings.GEMINI_API_KEY and not settings.CLAUDE_API_KEY,
    reason="No LLM keys configured at all — skipping",
)
def test_call_llm_strategy_produces_response():
    """
    Calls the unified call_llm() with the current LLM_PROVIDER_STRATEGY
    and verifies at least one provider returns a response.
    """
    try:
        from app.services.llm_client import call_llm
    except ImportError as e:
        pytest.skip(f"Missing dependency: {e}")

    result = call_llm(
        prompt="Reply with the single word: PONG",
        temperature=0.0,
        max_tokens=16,
        call_type="strategy_test",
    )

    error_str = str(result.get("error", "")).lower()
    is_quota_or_billing = any(
        k in error_str
        for k in ("429", "quota", "resource_exhausted", "prepayment", "credit", "billing")
    )

    if is_quota_or_billing:
        pytest.skip(
            f"Strategy={settings.LLM_PROVIDER_STRATEGY!r}: key valid but account has quota/billing issues"
        )

    assert result["success"], (
        f"call_llm() failed with strategy={settings.LLM_PROVIDER_STRATEGY!r}.\n"
        f"Error: {result.get('error')}\n"
        f"Make sure at least one of GEMINI_API_KEY / CLAUDE_API_KEY is set in .env"
    )
    assert result["response_text"].strip(), "call_llm() returned an empty response"
    print(
        f"\n✅ call_llm OK | strategy={settings.LLM_PROVIDER_STRATEGY!r} | "
        f"model={result['model']} | {result['latency_ms']}ms | "
        f"response='{result['response_text'].strip()}'"
    )
