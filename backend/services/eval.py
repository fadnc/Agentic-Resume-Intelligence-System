"""
eval.py

RAGAS-inspired retrieval quality metrics computed per inference.

Metrics:
  1. context_relevance   — how semantically similar are the retrieved chunks
                           to the query (job description)?  0–1 cosine score.

  2. faithfulness        — does the LLM output stay grounded in the retrieved
                           context, or does it introduce content not present?
                           Measured as token-overlap (Jaccard) between context
                           and LLM response.  0–1.

  3. answer_relevance    — does the LLM response address the query (JD)?
                           Cosine similarity between JD embedding and
                           response embedding.  0–1.

Each inference appends an EvalRecord to an in-memory log (capped at 200).
Aggregates are exposed via get_eval_summary().
"""

import time
import logging
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import Optional
from backend.services.embeddings import embed_texts

logger = logging.getLogger("resume-ai")

# ── data model ─────────────────────────────────────────────────────────────────

@dataclass
class EvalRecord:
    timestamp: float
    context_relevance: float          # retrieved chunks vs JD query
    faithfulness: float               # LLM output grounded in context?
    answer_relevance: float           # LLM output relevant to JD?
    resume_chunks_used: int
    kb_chunks_used: int
    llm_score: int                    # the 0-100 score the LLM returned
    latency_ms: Optional[float] = None

    def to_dict(self) -> dict:
        return asdict(self)


# ── in-memory log ──────────────────────────────────────────────────────────────

_eval_log: list[EvalRecord] = []
_MAX_LOG_SIZE = 200


def _append(record: EvalRecord) -> None:
    global _eval_log
    _eval_log.append(record)
    if len(_eval_log) > _MAX_LOG_SIZE:
        _eval_log = _eval_log[-_MAX_LOG_SIZE:]


# ── metric helpers ─────────────────────────────────────────────────────────────

def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D float32 vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def _mean_cosine(query_vec: np.ndarray, chunks: list[str]) -> float:
    """
    Average cosine similarity between query_vec and each chunk's embedding.
    Empty chunk list → 0.0.
    """
    if not chunks:
        return 0.0
    chunk_vecs = embed_texts(chunks)          # (n, dim) float32
    sims = [_cosine_similarity(query_vec, cv) for cv in chunk_vecs]
    return round(float(np.mean(sims)), 4)


def _jaccard_token_overlap(text_a: str, text_b: str) -> float:
    """
    Token-level Jaccard similarity.
    Proxy for faithfulness: how much of the response vocabulary
    appears in the context.
    """
    tokens_a = set(text_a.lower().split())
    tokens_b = set(text_b.lower().split())
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b
    return round(len(intersection) / len(union), 4)


# ── public API ─────────────────────────────────────────────────────────────────

def compute_and_log(
    job_text: str,
    job_vec: np.ndarray,
    resume_chunks: list[str],
    kb_chunks: list[str],
    llm_response: dict,
    latency_ms: float,
) -> EvalRecord:
    """
    Compute all three metrics and append to the eval log.

    Args:
        job_text       : raw job description string (used for faithfulness)
        job_vec        : pre-computed embedding of the job description
        resume_chunks  : top-k chunks retrieved from the resume index
        kb_chunks      : all chunks retrieved from knowledge base corpora
        llm_response   : the parsed dict returned by call_llm()
        latency_ms     : end-to-end pipeline latency

    Returns:
        EvalRecord
    """
    all_chunks = resume_chunks + kb_chunks

    # 1. Context relevance — how relevant are retrieved chunks to the JD?
    context_relevance = _mean_cosine(job_vec, all_chunks)

    # 2. Faithfulness — does LLM output stay in-context?
    # Flatten LLM response to a single string for overlap calculation
    response_text = " ".join([
        str(llm_response.get("score", "")),
        " ".join(llm_response.get("missing_skills", [])),
        " ".join(llm_response.get("suggestions", [])),
        " ".join(llm_response.get("rewritten_bullets", [])),
    ])
    context_text = " ".join(all_chunks)
    faithfulness = _jaccard_token_overlap(context_text, response_text)

    # 3. Answer relevance — is the LLM response relevant to the JD?
    response_vec = embed_texts([response_text])[0]
    answer_relevance = _cosine_similarity(job_vec, response_vec)
    answer_relevance = round(answer_relevance, 4)

    record = EvalRecord(
        timestamp=time.time(),
        context_relevance=context_relevance,
        faithfulness=faithfulness,
        answer_relevance=answer_relevance,
        resume_chunks_used=len(resume_chunks),
        kb_chunks_used=len(kb_chunks),
        llm_score=llm_response.get("score", 0),
        latency_ms=round(latency_ms, 2),
    )

    _append(record)
    logger.info(
        f"[eval] ctx_rel={context_relevance:.3f} "
        f"faithful={faithfulness:.3f} "
        f"ans_rel={answer_relevance:.3f} "
        f"latency={latency_ms:.0f}ms"
    )
    return record


def get_eval_summary() -> dict:
    """Aggregate stats over all logged eval records."""
    if not _eval_log:
        return {"total_evaluations": 0, "message": "No evaluations logged yet."}

    def _avg(key):
        vals = [getattr(r, key) for r in _eval_log if getattr(r, key) is not None]
        return round(float(np.mean(vals)), 4) if vals else None

    return {
        "total_evaluations": len(_eval_log),
        "avg_context_relevance": _avg("context_relevance"),
        "avg_faithfulness": _avg("faithfulness"),
        "avg_answer_relevance": _avg("answer_relevance"),
        "avg_latency_ms": _avg("latency_ms"),
        "avg_llm_score": _avg("llm_score"),
        "latest": _eval_log[-1].to_dict() if _eval_log else None,
    }


def get_eval_history(limit: int = 50) -> list[dict]:
    """Return the most recent `limit` eval records as dicts."""
    return [r.to_dict() for r in _eval_log[-limit:]]