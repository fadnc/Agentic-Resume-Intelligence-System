from fastapi import APIRouter, Query
from backend.services.eval import get_eval_summary, get_eval_history
from backend.services.corpus_builder import get_corpus_stats

router = APIRouter(prefix="/eval", tags=["eval"])


@router.get("/metrics")
def eval_metrics():
    """
    Aggregated RAGAS-style retrieval quality metrics across all logged inferences.

    Returns:
      - total_evaluations
      - avg_context_relevance  (0–1): retrieved chunks vs job description
      - avg_faithfulness       (0–1): LLM output grounded in context
      - avg_answer_relevance   (0–1): LLM output relevant to job description
      - avg_latency_ms
      - avg_llm_score          (0–100)
      - latest                 : most recent eval record
    """
    return get_eval_summary()


@router.get("/history")
def eval_history(limit: int = Query(default=50, ge=1, le=200)):
    """
    Per-inference eval records (most recent `limit` entries).
    Useful for plotting metric trends over time.
    """
    records = get_eval_history(limit=limit)
    return {
        "count": len(records),
        "records": records,
    }


@router.get("/corpus/stats")
def corpus_stats():
    """
    Stats for all loaded knowledge base corpora.
    Returns document count and index type per corpus.
    """
    stats = get_corpus_stats()
    if not stats:
        return {
            "status": "not_loaded",
            "message": "Knowledge base corpora not yet initialized. "
                       "Start the backend — corpora load on startup."
        }
    return {
        "status": "loaded",
        "corpora": stats,
        "total_docs": sum(v["num_docs"] for v in stats.values()),
    }