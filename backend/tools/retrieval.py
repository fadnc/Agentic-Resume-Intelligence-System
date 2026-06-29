"""
backend/tools/retrieval.py

Thin wrappers around v3 retriever.py and corpus_builder.py
so the LangGraph retrieval_node can call them cleanly.
"""
import numpy as np
from backend.services.retriever import create_index, dual_search
from backend.services.chunker import chunk_text
from backend.services.embeddings import embed_texts


def build_resume_index(resume_text: str) -> str:
    """
    Chunk and embed resume text, build per-request FAISS index.
    Returns the MD5 cache key.
    """
    chunks = chunk_text(resume_text, size=300, overlap=50)
    vectors = embed_texts(chunks)
    key = create_index(vectors, chunks, resume_text)
    return key


def dual_retrieve(
    jd_vec: np.ndarray,
    resume_key: str,
    resume_k: int = 3,
    kb_k: int = 2,
) -> tuple[list, dict]:
    """
    Run dual search: top-k resume chunks + top-k per KB corpus.
    Returns (resume_chunks, kb_chunks_dict).
    """
    resume_chunks, kb_results = dual_search(
        query_vec=jd_vec,
        resume_key=resume_key,
        resume_k=resume_k,
        kb_k_per_corpus=kb_k,
    )
    return resume_chunks, kb_results