from pydantic import BaseModel
from typing import List, Optional


class EvalMetadata(BaseModel):
    context_relevance: float
    faithfulness: float
    answer_relevance: float
    latency_ms: float
    resume_chunks_used: int
    kb_chunks_used: int


class AnalyzeResponse(BaseModel):
    score: int
    missing_skills: List[str]
    suggestions: List[str]
    rewritten_bullets: List[str]
    eval: Optional[EvalMetadata] = None