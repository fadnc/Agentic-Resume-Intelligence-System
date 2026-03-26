"""
eval.py

RAGAS-inspired retrieval quality metrics computed per inference.
Results are persisted to a local SQLite database so they survive
server restarts, re-deploys, and Render free-tier cold starts.

Metrics:
  1. context_relevance  — how semantically similar are the retrieved
                          chunks to the job description?  0–1 cosine.

  2. faithfulness       — does the LLM output stay grounded in the
                          retrieved context?  Jaccard token overlap. 0–1.
                          (TODO: replace with NLI entailment for accuracy)

  3. answer_relevance   — does the LLM response address the JD?
                          Cosine similarity between JD and response
                          embeddings.  0–1.

Database:
  A single SQLite file at EVAL_DB_PATH (default: eval_log.db at
  the project root).  The table is created automatically on first run.
  All reads/writes go through _get_conn() which opens a short-lived
  connection — safe for single-worker deployments (Render free tier,
  local uvicorn --workers 1).
"""

import time
import logging
import sqlite3
import numpy as np
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional

from backend.services.embeddings import embed_texts

logger = logging.getLogger("resume-ai")

# ── Database path ──────────────────────────────────────────────────────────────
# Sits at the project root so it persists across deploys if the filesystem is
# mounted (e.g. a Render disk, a Docker volume, or plain local dev).
EVAL_DB_PATH = Path("eval_log.db")

# Cap how many rows the history endpoint will ever return.
MAX_HISTORY = 200


# ── Schema ─────────────────────────────────────────────────────────────────────

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS eval_records (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp           REAL    NOT NULL,
    context_relevance   REAL    NOT NULL,
    faithfulness        REAL    NOT NULL,
    answer_relevance    REAL    NOT NULL,
    resume_chunks_used  INTEGER NOT NULL,
    kb_chunks_used      INTEGER NOT NULL,
    llm_score           INTEGER NOT NULL,
    latency_ms          REAL
);
"""


# ── Connection helper ──────────────────────────────────────────────────────────

def _get_conn() -> sqlite3.Connection:
    """
    Open (or create) the SQLite database and ensure the table exists.
    Returns a connection with row_factory set so rows behave like dicts.
    Caller is responsible for closing.
    """
    conn = sqlite3.connect(str(EVAL_DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute(_CREATE_TABLE)
    conn.commit()
    return conn


# ── Data model ─────────────────────────────────────────────────────────────────

@dataclass
class EvalRecord:
    timestamp:           float
    context_relevance:   float
    faithfulness:        float
    answer_relevance:    float
    resume_chunks_used:  int
    kb_chunks_used:      int
    llm_score:           int
    latency_ms:          Optional[float] = None

    def to_dict(self) -> dict:
        return asdict(self)


# ── Metric helpers ─────────────────────────────────────────────────────────────

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
    chunk_vecs = embed_texts(chunks)
    sims = [_cosine_similarity(query_vec, cv) for cv in chunk_vecs]
    return round(float(np.mean(sims)), 4)


def _jaccard_token_overlap(text_a: str, text_b: str) -> float:
    """
    Token-level Jaccard similarity.
    Proxy for faithfulness: how much of the response vocabulary
    appears in the context.
    NOTE: replace with NLI-based entailment for production accuracy.
    """
    tokens_a = set(text_a.lower().split())
    tokens_b = set(text_b.lower().split())
    if not tokens_a or not tokens_b:
        return 0.0
    return round(len(tokens_a & tokens_b) / len(tokens_a | tokens_b), 4)


# ── Write ──────────────────────────────────────────────────────────────────────

def _insert(record: EvalRecord) -> None:
    """Persist one EvalRecord to SQLite."""
    conn = _get_conn()
    try:
        conn.execute(
            """
            INSERT INTO eval_records
                (timestamp, context_relevance, faithfulness, answer_relevance,
                 resume_chunks_used, kb_chunks_used, llm_score, latency_ms)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.timestamp,
                record.context_relevance,
                record.faithfulness,
                record.answer_relevance,
                record.resume_chunks_used,
                record.kb_chunks_used,
                record.llm_score,
                record.latency_ms,
            ),
        )
        conn.commit()
    finally:
        conn.close()


# ── Public API ─────────────────────────────────────────────────────────────────

def compute_and_log(
    job_text: str,
    job_vec: np.ndarray,
    resume_chunks: list[str],
    kb_chunks: list[str],
    llm_response: dict,
    latency_ms: float,
) -> EvalRecord:
    """
    Compute all three metrics, persist to SQLite, and return the record.

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

    # 1. Context relevance
    context_relevance = _mean_cosine(job_vec, all_chunks)

    # 2. Faithfulness (Jaccard proxy)
    response_text = " ".join([
        str(llm_response.get("score", "")),
        " ".join(llm_response.get("missing_skills", [])),
        " ".join(llm_response.get("suggestions", [])),
        " ".join(llm_response.get("rewritten_bullets", [])),
    ])
    context_text = " ".join(all_chunks)
    faithfulness = _jaccard_token_overlap(context_text, response_text)

    # 3. Answer relevance
    response_vec     = embed_texts([response_text])[0]
    answer_relevance = round(_cosine_similarity(job_vec, response_vec), 4)

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

    _insert(record)

    logger.info(
        f"[eval] ctx_rel={context_relevance:.3f} "
        f"faithful={faithfulness:.3f} "
        f"ans_rel={answer_relevance:.3f} "
        f"latency={latency_ms:.0f}ms "
        f"[persisted to {EVAL_DB_PATH}]"
    )
    return record


def get_eval_summary() -> dict:
    """
    Aggregate stats across ALL persisted eval records.
    This now reflects the full history, not just the current process lifetime.
    """
    conn = _get_conn()
    try:
        row = conn.execute(
            """
            SELECT
                COUNT(*)                        AS total,
                AVG(context_relevance)          AS avg_ctx,
                AVG(faithfulness)               AS avg_faith,
                AVG(answer_relevance)           AS avg_ans,
                AVG(latency_ms)                 AS avg_lat,
                AVG(llm_score)                  AS avg_score
            FROM eval_records
            """
        ).fetchone()

        total = row["total"]
        if total == 0:
            return {"total_evaluations": 0, "message": "No evaluations logged yet."}

        # Fetch the most recent record for the "latest" field
        latest_row = conn.execute(
            """
            SELECT * FROM eval_records ORDER BY timestamp DESC LIMIT 1
            """
        ).fetchone()

        latest = dict(latest_row) if latest_row else None

        return {
            "total_evaluations":    total,
            "avg_context_relevance": round(row["avg_ctx"],   4) if row["avg_ctx"]   is not None else None,
            "avg_faithfulness":      round(row["avg_faith"], 4) if row["avg_faith"] is not None else None,
            "avg_answer_relevance":  round(row["avg_ans"],   4) if row["avg_ans"]   is not None else None,
            "avg_latency_ms":        round(row["avg_lat"],   2) if row["avg_lat"]   is not None else None,
            "avg_llm_score":         round(row["avg_score"], 4) if row["avg_score"] is not None else None,
            "latest": latest,
        }
    finally:
        conn.close()


def get_eval_history(limit: int = 50) -> list[dict]:
    """
    Return the most recent `limit` eval records as dicts, newest first.
    Capped at MAX_HISTORY to keep response sizes sane.
    """
    limit = min(limit, MAX_HISTORY)
    conn = _get_conn()
    try:
        rows = conn.execute(
            """
            SELECT * FROM eval_records
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()