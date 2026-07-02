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
    client = Groq(api_key=GROQ_API_KEY)

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
        response_format={"type": "json_object"},
    )
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