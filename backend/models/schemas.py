"""
backend/models/schemas.py — v7
Extends v3 AnalyzeResponse with v7 agent output fields.
Fully backward-compatible: v3 fields (score, missing_skills, suggestions,
rewritten_bullets, eval) are unchanged.
"""
from pydantic import BaseModel
from typing import List, Optional, Any


class EvalMetadata(BaseModel):
    """v3 RAGAS-style metrics — unchanged."""
    context_relevance:  float
    faithfulness:       float
    answer_relevance:   float
    latency_ms:         float
    resume_chunks_used: int
    kb_chunks_used:     int


class AnalyzeResponse(BaseModel):
    # ── v3 fields (unchanged) ─────────────────────────────────────────────────
    score:             int
    missing_skills:    List[str]
    suggestions:       List[str]
    rewritten_bullets: List[str]
    eval:              Optional[Any] = None   # EvalMetadata dict or None

    # ── v7 additions ──────────────────────────────────────────────────────────
    matched_skills:    List[str] = []
    gap_summary:       str = ""
    tailored_summary:  str = ""
    quick_wins:        List[str] = []
    agent_trace:       List[str] = []         # per-agent log messages
    processing_time_ms: float = 0.0