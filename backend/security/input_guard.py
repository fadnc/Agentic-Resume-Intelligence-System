"""Bound and screen untrusted text before it reaches agent prompts."""
import re


MAX_RESUME_TEXT_CHARS = 50_000
MAX_JOB_DESCRIPTION_CHARS = 30_000
MAX_PDF_BYTES = 4 * 1024 * 1024

_INJECTION_PATTERNS = (
    re.compile(r"\bignore\s+(all\s+)?(previous|prior|above|earlier)\s+instructions\b", re.I),
    re.compile(r"\b(disregard|forget)\s+(all\s+)?(previous|prior|above|earlier)?\s*(instructions|rules|prompt)\b", re.I),
    re.compile(r"\b(reveal|print|show|output|expose)\s+(the\s+)?(system|developer|hidden)\s+(prompt|instructions|message)\b", re.I),
    re.compile(r"\b(reveal|print|show|output|expose)\s+(the\s+)?(api\s+key|credentials|secrets)\b", re.I),
    re.compile(r"\b(override|bypass|disable)\s+(the\s+)?(safety|security|system|developer)\s+(rules|instructions|prompt|filters?)\b", re.I),
    re.compile(r"\b(act|behave)\s+as\s+(the\s+)?(system|developer|unrestricted|unfiltered)\b", re.I),
    re.compile(r"\b(jailbreak|DAN\s+mode|do anything now)\b", re.I),
)


def check_untrusted_text(text: str, *, label: str, max_chars: int) -> str:
    """Validate text bounds and reject common direct prompt-injection commands."""
    if not isinstance(text, str):
        raise ValueError(f"{label} must be text.")
    cleaned = text.replace("\x00", "").strip()
    if not cleaned:
        raise ValueError(f"{label} is empty.")
    if len(cleaned) > max_chars:
        raise ValueError(f"{label} is too long (maximum {max_chars:,} characters).")
    if any(pattern.search(cleaned) for pattern in _INJECTION_PATTERNS):
        raise ValueError(f"{label} contains instructions that are not allowed. Remove them and try again.")
    return cleaned


def untrusted_block(text: str) -> str:
    """Encode document text as data so it cannot alter the surrounding prompt."""
    import json

    encoded = json.dumps(text, ensure_ascii=False)
    encoded = encoded.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    return "<untrusted_document_json>\n" + encoded + "\n</untrusted_document_json>"
