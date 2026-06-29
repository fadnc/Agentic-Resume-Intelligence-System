"""
ResumeState — shared TypedDict for the LangGraph agent graph.

Combines:
- v3 fields: resume_text, jd_text, retrieved chunks, KB context, eval metadata
- v7 fields: parsed_resume, parsed_jd, match_score, rewritten bullets, agent trace
"""
from typing import TypedDict, Annotated, Optional
from operator import add


class ResumeState(TypedDict):
    # ── Inputs ────────────────────────────────────────────────────────────────
    resume_text: str                    # extracted from PDF upstream
    jd_text: str                        # raw job description
    jd_url: Optional[str]               # if user pasted a URL instead

    # ── Retrieval (v3 dual-retrieval carried forward) ─────────────────────────
    resume_chunks: Optional[list]       # top-k chunks from resume FAISS index
    kb_chunks: Optional[dict]           # {corpus_name: [chunk, ...]} from 4 KB corpora
    resume_index_key: Optional[str]     # MD5 key into retriever cache

    # ── Parser agent ──────────────────────────────────────────────────────────
    parsed_resume: Optional[dict]       # {name, skills, experience, education, summary}

    # ── JD analyst agent ──────────────────────────────────────────────────────
    parsed_jd: Optional[dict]           # {title, company, must_have, nice_to_have, keywords, domain}

    # ── Gap analyst agent ─────────────────────────────────────────────────────
    match_score: Optional[int]          # 0–100
    matched_skills: Optional[list]
    missing_skills: Optional[list]      # [{skill, importance, learnable_in_weeks}]
    gap_summary: Optional[str]
    quick_wins: Optional[list]

    # ── Rewriter agent ────────────────────────────────────────────────────────
    rewritten_bullets: Optional[list]   # [{section, original, rewritten, reason}]
    tailored_summary: Optional[str]
    improvement_tips: Optional[list]

    # ── Eval (v3 RAGAS-style, computed after rewriter) ────────────────────────
    eval_metadata: Optional[dict]       # {context_relevance, faithfulness, answer_relevance, ...}

    # ── Trace / error ─────────────────────────────────────────────────────────
    messages: Annotated[list, add]      # append-only agent log
    error: Optional[str]
    latency_ms: Optional[float]