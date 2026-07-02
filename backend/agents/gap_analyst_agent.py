"""
Gap Analyst Agent
Combines semantic similarity (v3 FAISS embeddings) + LLM gap reasoning.
Grounds scoring in KB hiring_criteria corpus (v3).
"""
import json
import re
import numpy as np
from backend.agents.state import ResumeState
from backend.services.llm import call_llm
from backend.services.embeddings import embed_texts

GAP_PROMPT = """
You are a career coach doing a deep skills gap analysis.

Candidate skills: {resume_skills}
JD must-have requirements: {must_have}
JD nice-to-have: {nice_to_have}
ATS keywords: {keywords}
Semantic similarity score (0–1): {semantic_score:.3f}

Hiring criteria context (from knowledge base):
{hiring_criteria_context}

Return ONLY valid JSON:
{{
  "match_score": <integer 0-100>,
  "matched_skills": ["skills the candidate clearly has"],
  "missing_skills": [
    {{
      "skill": "name",
      "importance": "critical | important | nice_to_have",
      "learnable_in_weeks": <integer or null>
    }}
  ],
  "gap_summary": "2-3 sentence honest assessment",
  "quick_wins": ["things candidate can add to resume NOW without learning anything new"]
}}

Scoring guide:
80-100: Strong fit  |  60-79: Good fit, minor gaps
40-59: Moderate fit  |  20-39: Weak fit  |  0-19: Poor match
"""


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(np.dot(a, b) / (na * nb)) if na and nb else 0.0


def gap_analyst_agent(state: ResumeState) -> ResumeState:
    parsed_resume = state.get("parsed_resume") or {}
    parsed_jd = state.get("parsed_jd") or {}

    if not parsed_resume or not parsed_jd:
        return {**state, "error": "Missing parsed resume or JD", "messages": ["[gap_analyst] missing inputs"]}

    resume_skills = parsed_resume.get("skills", [])
    must_have     = parsed_jd.get("must_have", [])
    nice_to_have  = parsed_jd.get("nice_to_have", [])
    keywords      = parsed_jd.get("keywords", [])

    # Semantic similarity via v3 embeddings service
    try:
        vecs = embed_texts([" ".join(resume_skills), " ".join(must_have + keywords)])
        semantic_score = _cosine(vecs[0], vecs[1])
    except Exception:
        semantic_score = 0.5

    kb = state.get("kb_chunks") or {}
    hiring_ctx = "\n".join(kb.get("hiring_criteria", [])[:2])

    prompt = GAP_PROMPT.format(
        resume_skills=json.dumps(resume_skills),
        must_have=json.dumps(must_have),
        nice_to_have=json.dumps(nice_to_have),
        keywords=json.dumps(keywords[:15]),
        semantic_score=semantic_score,
        hiring_criteria_context=hiring_ctx or "Not available",
    )
    raw = call_llm(prompt, system="You are a JSON-only gap analyst.", temperature=0.1)
    raw = re.sub(r"```(?:json)?|```", "", raw).strip()

    try:
        analysis = json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        analysis = json.loads(m.group()) if m else None

    if not analysis:
        return {**state, "error": "Gap analyst JSON error", "messages": ["[gap_analyst] parse failed"]}

    score = analysis.get("match_score", 0)
    return {
        **state,
        "match_score":    score,
        "matched_skills": analysis.get("matched_skills", []),
        "missing_skills": analysis.get("missing_skills", []),
        "gap_summary":    analysis.get("gap_summary", ""),
        "quick_wins":     analysis.get("quick_wins", []),
        "messages": [
            f"[gap_analyst] score={score}/100 "
            f"gaps={len(analysis.get('missing_skills', []))} "
            f"semantic={semantic_score:.2f}"
        ],
    }