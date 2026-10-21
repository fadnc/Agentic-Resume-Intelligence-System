"""Helpers for parsing JSON returned by language models."""
import json
import re
from typing import Any, Optional


def parse_json_object(raw: str) -> Optional[dict[str, Any]]:
    """Parse a JSON object, including one wrapped in a Markdown code fence."""
    if not isinstance(raw, str) or not raw.strip():
        return None

    cleaned = re.sub(r"```(?:json)?\s*|```", "", raw, flags=re.IGNORECASE).strip()
    try:
        value = json.loads(cleaned)
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        pass

    # Model responses sometimes include a short preamble or trailing text.
    start = cleaned.find("{")
    if start < 0:
        return None
    try:
        value, _ = json.JSONDecoder().raw_decode(cleaned[start:])
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None
