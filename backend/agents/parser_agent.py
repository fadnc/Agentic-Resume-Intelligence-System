"""
Parser Agent
Extracts structured fields from resume text.
Uses resume_chunks (already retrieved) for focused context — avoids re-embedding.
"""
import json
import re
from backend.agents.state import ResumeState
from backend.services.llm import call_llm

PARSER_PROMPT = """
You are a precise resume parser. Extract structured information from the resume.

Resume text:
---
{resume_text}
---

Supporting excerpts (from FAISS retrieval — use for context):
---
{resume_chunks}
---

Return ONLY valid JSON, no markdown fences:
{{
  "name": "string",
  "email": "string or null",
  "skills": ["list of specific skills"],
  "experience": [
    {{
      "title": "string",
      "company": "string",
      "duration": "string",
      "bullets": ["bullet points"]
    }}
  ],
  "education": [
    {{
      "degree": "string",
      "institution": "string",
      "year": "string or null",
      "gpa": "string or null"
    }}
  ],
  "summary": "string or null",
  "certifications": ["list or empty array"]
}}
"""


def parser_agent(state: ResumeState) -> ResumeState:
    resume_text = state.get("resume_text", "")
    if not resume_text:
        return {**state, "error": "No resume text", "messages": ["[parser] no input"]}

    chunks_preview = "\n".join((state.get("resume_chunks") or [])[:3])

    prompt = PARSER_PROMPT.format(
        resume_text=resume_text[:3000],
        resume_chunks=chunks_preview,
    )
    raw = call_llm(prompt, system="You are a JSON-only resume parser.", temperature=0.1)
    raw = re.sub(r"```(?:json)?|```", "", raw).strip()

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        parsed = json.loads(m.group()) if m else None

    if not parsed:
        return {**state, "error": "Parser JSON error", "messages": ["[parser] parse failed"]}

    return {
        **state,
        "parsed_resume": parsed,
        "messages": [f"[parser] {len(parsed.get('skills', []))} skills, {len(parsed.get('experience', []))} roles"],
    }