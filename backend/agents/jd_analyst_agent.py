"""
JD Analyst Agent
Parses and ranks job description requirements.
Grounds analysis in the v3 KB corpora: job_market + ats_keywords.
"""
import json
import re
from backend.agents.state import ResumeState
from backend.services.llm import call_llm

JD_PROMPT = """
You are an expert technical recruiter analyzing a job description.

Job Description:
---
{jd_text}
---

Industry context from knowledge base:
[Job market standards]
{job_market_context}

[ATS keywords for this domain]
{ats_keyword_context}

Return ONLY valid JSON:
{{
  "title": "string",
  "company": "string or null",
  "seniority": "junior | mid | senior | staff | unknown",
  "domain": "string (e.g. ML Engineering, Backend, Data Science)",
  "must_have": ["explicitly required skills/qualifications"],
  "nice_to_have": ["preferred or bonus skills"],
  "keywords": ["important ATS keywords — exact tool/tech names"],
  "responsibilities": ["key responsibilities as brief phrases"]
}}
"""


def jd_analyst_agent(state: ResumeState) -> ResumeState:
    jd_text = state.get("jd_text", "")
    if not jd_text:
        return {**state, "error": "No JD text", "messages": ["[jd_analyst] no input"]}

    kb = state.get("kb_chunks") or {}
    job_market_ctx  = "\n".join(kb.get("job_market", [])[:2])
    ats_keyword_ctx = "\n".join(kb.get("ats_keywords", [])[:2])

    prompt = JD_PROMPT.format(
        jd_text=jd_text[:2000],
        job_market_context=job_market_ctx or "Not available",
        ats_keyword_context=ats_keyword_ctx or "Not available",
    )
    raw = call_llm(prompt, system="You are a JSON-only JD analyst.", temperature=0.1)
    raw = re.sub(r"```(?:json)?|```", "", raw).strip()

    try:
        parsed_jd = json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        parsed_jd = json.loads(m.group()) if m else None

    if not parsed_jd:
        return {**state, "error": "JD analyst JSON error", "messages": ["[jd_analyst] parse failed"]}

    return {
        **state,
        "parsed_jd": parsed_jd,
        "messages": [
            f"[jd_analyst] {len(parsed_jd.get('must_have', []))} must-have, "
            f"{len(parsed_jd.get('nice_to_have', []))} nice-to-have, "
            f"domain={parsed_jd.get('domain', '?')}"
        ],
    }