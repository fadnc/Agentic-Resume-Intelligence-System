"""
pipeline.py

Main analysis workflow.

Flow:
  1. Parse PDF → extract resume text
  2. Chunk + embed resume
  3. Build per-request resume FAISS index
  4. Dual retrieval: top-k resume chunks + top-k KB chunks per corpus
  5. Build prompt with merged context (resume + 4 KB corpora)
  6. Call LLM
  7. Compute + log RAGAS-style eval metrics
  8. Return result + eval metadata
"""

import time
import logging
from functools import lru_cache

from backend.services.parser import extract_text_from_pdf
from backend.services.chunker import chunk_text
from backend.services.embeddings import embed_texts
from backend.services.retriever import create_index, dual_search
from backend.services.llm import call_llm
from backend.services.eval import compute_and_log
from backend.models.prompts import PROMPT_TEMPLATE

logger = logging.getLogger("resume-ai")

# Context budget (characters) per source — keeps total prompt manageable
_RESUME_CTX_LIMIT   = 1500
_JD_LIMIT           = 1000
_KB_CTX_PER_CORPUS  = 600   # per corpus → 4 × 600 = 2400 chars of KB context


@lru_cache(maxsize=100)
def _get_jd_embedding(job_text: str):
    """Cache JD embeddings — same JD often reused across sessions."""
    return embed_texts([job_text])[0]


async def analyze_resume(resume_file, job_text: str) -> dict:
    t0 = time.time()
    logger.info("[pipeline] Starting analysis")

    # ── 1. Parse PDF ──────────────────────────────────────────────────────────
    file_bytes = await resume_file.read()
    resume_text = await extract_text_from_pdf(file_bytes)

    # ── 2. Chunk + embed resume ───────────────────────────────────────────────
    chunks = chunk_text(resume_text, size=200, overlap=30)
    vectors = embed_texts(chunks)

    # ── 3. Build resume index ─────────────────────────────────────────────────
    cache_key = create_index(vectors, chunks, resume_text)

    # ── 4. Dual retrieval ─────────────────────────────────────────────────────
    job_vec = _get_jd_embedding(job_text)

    resume_chunks, kb_results = dual_search(
        query_vec=job_vec,
        resume_key=cache_key,
        resume_k=3,
        kb_k_per_corpus=2,
    )

    # ── 5. Build prompt with merged context ───────────────────────────────────
    resume_context = "\n".join(resume_chunks)[:_RESUME_CTX_LIMIT]
    jd_truncated   = job_text[:_JD_LIMIT]

    def _kb_ctx(corpus_name: str) -> str:
        chunks = kb_results.get(corpus_name, [])
        return "\n".join(chunks)[:_KB_CTX_PER_CORPUS] if chunks else "N/A"

    prompt = PROMPT_TEMPLATE.format(
        resume=resume_context,
        jd=jd_truncated,
        job_market_context=_kb_ctx("job_market"),
        resume_example_context=_kb_ctx("resume_examples"),
        ats_keyword_context=_kb_ctx("ats_keywords"),
        hiring_criteria_context=_kb_ctx("hiring_criteria"),
    )

    logger.debug(f"[pipeline] Prompt length: {len(prompt)} chars")

    # ── 6. Call LLM ───────────────────────────────────────────────────────────
    result = call_llm(prompt)

    # ── 7. Eval metrics ───────────────────────────────────────────────────────
    latency_ms = (time.time() - t0) * 1000
    all_kb_chunks = [c for chunks in kb_results.values() for c in chunks]

    eval_record = compute_and_log(
        job_text=jd_truncated,
        job_vec=job_vec,
        resume_chunks=resume_chunks,
        kb_chunks=all_kb_chunks,
        llm_response=result,
        latency_ms=latency_ms,
    )

    # ── 8. Return result + eval metadata ──────────────────────────────────────
    result["eval"] = {
        "context_relevance": eval_record.context_relevance,
        "faithfulness":      eval_record.faithfulness,
        "answer_relevance":  eval_record.answer_relevance,
        "latency_ms":        eval_record.latency_ms,
        "resume_chunks_used": eval_record.resume_chunks_used,
        "kb_chunks_used":     eval_record.kb_chunks_used,
    }

    logger.info(
        f"[pipeline] Done in {latency_ms:.0f}ms | "
        f"score={result.get('score')} | "
        f"ctx_rel={eval_record.context_relevance:.3f}"
    )
    return result