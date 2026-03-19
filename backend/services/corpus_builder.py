"""
corpus_builder.py

Builds and manages the persistent multi-corpus RAG knowledge base.
Four corpora, each stored as a separate FAISS index + metadata file:
  - job_market       : role descriptions, responsibilities, expectations
  - resume_examples  : high-scoring resume bullets by domain/level
  - ats_keywords     : keyword lists per role and tech stack
  - hiring_criteria  : rubrics, ATS scoring logic, interview signals

Each corpus is loaded once at startup and cached in memory.
"""

import os
import json
import faiss
import numpy as np
import threading
import logging
from pathlib import Path
from backend.services.embeddings import embed_texts

logger = logging.getLogger("resume-ai")

# ── paths ──────────────────────────────────────────────────────────────────────

KB_DIR = Path("knowledge_base")
KB_STORE_DIR = Path("embeddings_store/knowledge_base")

CORPUS_CONFIGS = {
    "job_market": {
        "source": KB_DIR / "job_market.json",
        "index_path": KB_STORE_DIR / "job_market.faiss",
        "meta_path":  KB_STORE_DIR / "job_market_meta.npy",
    },
    "resume_examples": {
        "source": KB_DIR / "resume_examples.json",
        "index_path": KB_STORE_DIR / "resume_examples.faiss",
        "meta_path":  KB_STORE_DIR / "resume_examples_meta.npy",
    },
    "ats_keywords": {
        "source": KB_DIR / "ats_keywords.json",
        "index_path": KB_STORE_DIR / "ats_keywords.faiss",
        "meta_path":  KB_STORE_DIR / "ats_keywords_meta.npy",
    },
    "hiring_criteria": {
        "source": KB_DIR / "hiring_criteria.json",
        "index_path": KB_STORE_DIR / "hiring_criteria.faiss",
        "meta_path":  KB_STORE_DIR / "hiring_criteria_meta.npy",
    },
}

# ── in-memory cache ────────────────────────────────────────────────────────────

_KB_CACHE: dict[str, tuple[faiss.Index, list[str]]] = {}
_lock = threading.Lock()


# ── build / load ───────────────────────────────────────────────────────────────

def _build_corpus(name: str, cfg: dict) -> tuple[faiss.Index, list[str]]:
    """Embed all documents in a corpus JSON and write FAISS index + meta."""
    logger.info(f"[corpus_builder] Building corpus: {name}")

    with open(cfg["source"], "r") as f:
        documents = json.load(f)

    texts = [doc["content"] for doc in documents]
    vectors = embed_texts(texts)                     # float32, normalised

    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)                   # inner-product = cosine on normalised vecs
    index.add(vectors)

    # persist
    cfg["index_path"].parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(cfg["index_path"]))
    np.save(str(cfg["meta_path"]), np.array(texts, dtype=object))

    logger.info(f"[corpus_builder] Built {name}: {len(texts)} docs, dim={dim}")
    return index, texts


def _load_corpus(name: str, cfg: dict) -> tuple[faiss.Index, list[str]]:
    """Load a pre-built FAISS index + metadata from disk."""
    logger.info(f"[corpus_builder] Loading corpus from disk: {name}")
    index = faiss.read_index(str(cfg["index_path"]))
    texts = np.load(str(cfg["meta_path"]), allow_pickle=True).tolist()
    return index, texts


def _ensure_corpus(name: str, cfg: dict) -> tuple[faiss.Index, list[str]]:
    """Return (index, texts), building from source if not already on disk."""
    index_exists = cfg["index_path"].exists()
    meta_exists  = cfg["meta_path"].exists()

    if index_exists and meta_exists:
        return _load_corpus(name, cfg)
    else:
        return _build_corpus(name, cfg)


def load_all_corpora(force_rebuild: bool = False) -> None:
    """
    Load all 4 corpora into _KB_CACHE.
    Call once at application startup (e.g. FastAPI lifespan).
    Set force_rebuild=True to re-embed from source JSONs.
    """
    with _lock:
        for name, cfg in CORPUS_CONFIGS.items():
            if name in _KB_CACHE and not force_rebuild:
                continue
            if force_rebuild and cfg["index_path"].exists():
                cfg["index_path"].unlink()
                cfg["meta_path"].unlink()
            _KB_CACHE[name] = _ensure_corpus(name, cfg)

    logger.info(f"[corpus_builder] All corpora loaded: {list(_KB_CACHE.keys())}")


# ── query ──────────────────────────────────────────────────────────────────────

def search_corpus(
    corpus_name: str,
    query_vector: np.ndarray,
    k: int = 2
) -> list[str]:
    """
    Return top-k text chunks from a named corpus for a given query vector.
    query_vector must be float32, shape (dim,) — already normalised.
    """
    if corpus_name not in _KB_CACHE:
        logger.warning(f"[corpus_builder] Corpus '{corpus_name}' not loaded — skipping")
        return []

    index, texts = _KB_CACHE[corpus_name]
    k = min(k, len(texts))
    scores, indices = index.search(query_vector.reshape(1, -1), k)
    return [texts[i] for i in indices[0] if i < len(texts)]


def search_all_corpora(
    query_vector: np.ndarray,
    k_per_corpus: int = 2
) -> dict[str, list[str]]:
    """
    Search all 4 corpora and return results keyed by corpus name.
    """
    return {
        name: search_corpus(name, query_vector, k=k_per_corpus)
        for name in _KB_CACHE
    }


def get_corpus_stats() -> dict:
    """Return size info for all loaded corpora (useful for /health and /eval endpoints)."""
    return {
        name: {
            "num_docs": len(texts),
            "index_type": type(index).__name__,
        }
        for name, (index, texts) in _KB_CACHE.items()
    }