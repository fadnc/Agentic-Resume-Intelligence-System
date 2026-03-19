"""
retriever.py

Dual retrieval:
  1. Resume index   — per-request FAISS index built from the uploaded resume
  2. Knowledge base — persistent multi-corpus index via corpus_builder

Resume indexes are keyed by MD5 hash of resume text and cached in memory.
"""

import faiss
import hashlib
import threading
import numpy as np
import logging
from backend.services.corpus_builder import search_all_corpora

logger = logging.getLogger("resume-ai")

# ── resume index cache ─────────────────────────────────────────────────────────

_INDEX_CACHE: dict[str, tuple[faiss.Index, list[str]]] = {}
_lock = threading.Lock()


def hash_text(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()


def create_index(vectors: np.ndarray, chunks: list[str], resume_text: str) -> str:
    """
    Build (or retrieve cached) FAISS index for a resume.
    Returns the cache key (MD5 of resume text).
    """
    key = hash_text(resume_text)

    with _lock:
        if key in _INDEX_CACHE:
            return key

        dim = vectors.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(vectors)
        _INDEX_CACHE[key] = (index, chunks)
        logger.debug(f"[retriever] Created resume index key={key[:8]} chunks={len(chunks)}")

    return key


def search_resume(
    query_vec: np.ndarray,
    key: str,
    k: int = 3
) -> list[str]:
    """Top-k chunks from the resume index."""
    index, chunks = _INDEX_CACHE[key]
    k = min(k, len(chunks))
    distances, indices = index.search(query_vec.reshape(1, -1), k)
    return [chunks[i] for i in indices[0]]


def search_knowledge_base(
    query_vec: np.ndarray,
    k_per_corpus: int = 2
) -> dict[str, list[str]]:
    """
    Top-k chunks from each of the 4 knowledge base corpora.
    Returns dict keyed by corpus name.
    """
    return search_all_corpora(query_vec, k_per_corpus=k_per_corpus)


def dual_search(
    query_vec: np.ndarray,
    resume_key: str,
    resume_k: int = 3,
    kb_k_per_corpus: int = 2,
) -> tuple[list[str], dict[str, list[str]]]:
    """
    Perform both retrieval sources in one call.

    Returns:
        resume_chunks : list of top-k resume text chunks
        kb_results    : dict of {corpus_name: [chunk, ...]}
    """
    resume_chunks = search_resume(query_vec, resume_key, k=resume_k)
    kb_results    = search_knowledge_base(query_vec, k_per_corpus=kb_k_per_corpus)

    total_kb = sum(len(v) for v in kb_results.values())
    logger.debug(
        f"[retriever] dual_search: resume={len(resume_chunks)} chunks, "
        f"kb={total_kb} chunks across {len(kb_results)} corpora"
    )
    return resume_chunks, kb_results