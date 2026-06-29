"""
Rewriter Agent
Rewrites resume bullets and summary using JD language.
Grounds output in v3 KB: resume_examples + hiring_criteria corpora.
Never fabricates experience.
"""
import json
import re
from backend.agents.state import ResumeState
from backend.services.llm import call_llm

REWRITER_PROMPT = """
You are an expert resume writer. Rewrite resume content to match a specific job.

Candidate experience (top roles):
{experience}

Target role: {job_title}
Must-have skills: {must_have}
ATS keywords to weave in: {keywords}
Already matched (don't over-emphasise): {matched}
Gaps (DO NOT fabricate): {missing_names}

High-scoring resume examples from knowledge base:
{resume_example_context}

Hiring criteria context:
{hiring_criteria_context}

Rules:
- Only rewrite bullets that can genuinely improve with better phrasing
- DO NOT invent experience, metrics, or skills the candidate doesn't have
- Mirror JD language exactly (if JD says "deployed", use "deployed")
- Strong action verbs, quantify where original already had numbers
- Max 2 lines per bullet

Return ONLY valid JSON:
{{
  "rewritten_bullets": [
    {{
      "section": "title at company",
      "original": "original bullet",
      "rewritten": "improved bullet",
      "reason": "why this helps"
    }}
  ],
  "tailored_summary": "3-sentence summary targeting this specific role",
  "improvement_tips": [
    "Specific actionable tip (course, cert, project to add)"
  ]
}}
"""


def rewriter_agent(state: ResumeState) -> ResumeState:
    parsed_resume = state.get("parsed_resume") or {}
    parsed_jd     = state.get("parsed_jd") or {}
    matched       = state.get("matched_skills") or []
    missing       = state.get("missing_skills") or []

    experience = parsed_resume.get("experience", [])[:3]
    missing_names = [m["skill"] if isinstance(m, dict) else m for m in missing]

    kb = state.get("kb_chunks") or {}
    resume_example_ctx  = "\n".join(kb.get("resume_examples", [])[:2])
    hiring_criteria_ctx = "\n".join(kb.get("hiring_criteria", [])[:1])

    prompt = REWRITER_PROMPT.format(
        experience=json.dumps(experience, indent=2),
        job_title=parsed_jd.get("title", "the role"),
        must_have=json.dumps(parsed_jd.get("must_have", [])),
        keywords=json.dumps(parsed_jd.get("keywords", [])[:15]),
        matched=json.dumps(matched[:10]),
        missing_names=json.dumps(missing_names[:8]),
        resume_example_context=resume_example_ctx or "Not available",
        hiring_criteria_context=hiring_criteria_ctx or "Not available",
    )
    raw = call_llm(
        prompt,
        system="You are a JSON-only resume rewriter. Never fabricate experience.",
        temperature=0.2,
        max_tokens=1500,
    )
    raw = re.sub(r"```(?:json)?|```", "", raw).strip()

    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        result = json.loads(m.group()) if m else None

    if not result:
        return {**state, "error": "Rewriter JSON error", "messages": ["[rewriter] parse failed"]}

    bullets = result.get("rewritten_bullets", [])
    return {
        **state,
        "rewritten_bullets": bullets,
        "tailored_summary":  result.get("tailored_summary", ""),
        "improvement_tips":  result.get("improvement_tips", []),
        "messages": [f"[rewriter] {len(bullets)} bullets rewritten, tailored summary generated"],
    }