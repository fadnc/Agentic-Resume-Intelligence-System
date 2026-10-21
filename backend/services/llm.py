"""
backend/services/llm.py — v7
Groq dispatcher unchanged from v3.
Added: temperature and max_tokens params so agents can control output.
"""
import json
import logging
from backend.config import USE_SAGEMAKER, GROQ_API_KEY, USE_GROQ

logger = logging.getLogger("resume-ai")


def call_groq_llm(
    prompt: str,
    system: str = "You are a helpful assistant.",
    temperature: float = 0.3,
    max_tokens: int = 1000,
) -> str:
    """Call Groq API. Returns raw text string (agents handle JSON parsing)."""
    from groq import Groq
    # Do not let SDK backoff hold an analysis open for 18+ seconds on rate
    # limits; the route maps 429 to a clear, retryable API response.
    client = Groq(api_key=GROQ_API_KEY, max_retries=0)

    request = {
        "model": "openai/gpt-oss-120b",
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "max_completion_tokens": max_tokens,
        # GPT-OSS is a reasoning model. Hide the reasoning field so constrained
        # JSON decoding receives only the final answer, and keep reasoning bounded.
        "reasoning_effort": "low",
        "include_reasoning": False,
    }
    try:
        response = client.chat.completions.create(
            **request,
            response_format={"type": "json_object"},
        )
    except Exception as exc:
        response_obj = getattr(exc, "response", None)
        response_headers = getattr(response_obj, "headers", {}) or {}
        logger.warning(
            "[llm] Groq request failed status=%s code=%s request_id=%s",
            getattr(exc, "status_code", "unknown"),
            getattr(exc, "code", None) or type(exc).__name__,
            response_headers.get("x-request-id"),
        )
        # Groq occasionally returns json_validate_failed with an empty generation
        # for gpt-oss even when the prompt requests valid JSON. Retry once without
        # constrained decoding; the agent's JSON parser validates the returned text.
        error = getattr(exc, "body", None)
        error = error.get("error", {}) if isinstance(error, dict) else {}
        if not isinstance(error, dict) or error.get("code") != "json_validate_failed":
            raise
        logger.warning("[llm] Groq JSON constrained output failed; retrying once in prompt-only JSON mode.")
        request["messages"] = [
            {"role": "system", "content": f"{system} Return exactly one valid JSON object, with no markdown or surrounding text."},
            {"role": "user", "content": prompt},
        ]
        response = client.chat.completions.create(**request)
    output = response.choices[0].message.content
    logger.debug(f"[llm] Groq response preview: {output[:120]}")
    return output


def call_local_llm(prompt: str, **kwargs) -> str:
    logger.warning("[llm] Fallback: GROQ_API_KEY not set.")
    return json.dumps({
        "score": 0,
        "missing_skills": ["GROQ_API_KEY not configured"],
        "suggestions": ["Add GROQ_API_KEY to .env — get free key at console.groq.com"],
        "rewritten_bullets": [],
    })


def call_llm(
    prompt: str,
    system: str = "You are a helpful assistant.",
    temperature: float = 0.3,
    max_tokens: int = 1000,
) -> str:
    """Main dispatcher — agents call this."""
    if USE_SAGEMAKER:
        raise NotImplementedError("SageMaker routing not yet implemented for agent calls.")
    if USE_GROQ:
        return call_groq_llm(prompt, system=system, temperature=temperature, max_tokens=max_tokens)
    return call_local_llm(prompt)
