"""
LangGraph orchestrator — v7

Replaces backend/services/pipeline.py.

Graph flow:
  scrape_jd (optional)
    → retrieval          ← v3 dual-retrieval (resume FAISS + 4 KB corpora)
    → parser_agent       ← extracts structured resume fields
    → jd_analyst_agent   ← ranks JD requirements
    → gap_analyst_agent  ← scores match, identifies gaps
    → [conditional]
        score ≥ 30  → rewriter_agent → eval_node → END
        score < 30  →                  eval_node → END
"""
import time
from langgraph.graph import StateGraph, END

from backend.agents.state import ResumeState
from backend.agents.parser_agent import parser_agent
from backend.agents.jd_analyst_agent import jd_analyst_agent
from backend.agents.gap_analyst_agent import gap_analyst_agent
from backend.agents.rewriter_agent import rewriter_agent
from backend.tools.scraper import scrape_jd
from backend.tools.retrieval import build_resume_index, dual_retrieve
from backend.services.eval import compute_and_log
from backend.services.embeddings import embed_texts


# ── Node: scrape JD from URL ──────────────────────────────────────────────────

def scrape_node(state: ResumeState) -> ResumeState:
    if state.get("jd_url") and not state.get("jd_text"):
        try:
            jd_text = scrape_jd(state["jd_url"])
            return {**state, "jd_text": jd_text, "messages": ["[scraper] JD fetched from URL"]}
        except Exception as e:
            return {**state, "error": f"Scraper failed: {e}"}
    return state


# ── Node: dual retrieval (v3 logic, now a graph node) ────────────────────────

def retrieval_node(state: ResumeState) -> ResumeState:
    """
    Builds per-request resume FAISS index and runs dual search:
    top-3 resume chunks + top-2 per KB corpus.
    Carried forward from v3 retriever/corpus_builder.
    """
    try:
        resume_text = state["resume_text"]
        jd_text = state["jd_text"]

        index_key = build_resume_index(resume_text)
        jd_vec = embed_texts([jd_text])[0]
        resume_chunks, kb_chunks = dual_retrieve(jd_vec, index_key)

        total_kb = sum(len(v) for v in kb_chunks.values())
        return {
            **state,
            "resume_index_key": index_key,
            "resume_chunks": resume_chunks,
            "kb_chunks": kb_chunks,
            "messages": [f"[retrieval] {len(resume_chunks)} resume chunks, {total_kb} KB chunks"],
        }
    except Exception as e:
        return {**state, "error": f"Retrieval failed: {e}"}


# ── Node: eval (v3 RAGAS metrics, now post-graph) ────────────────────────────

def eval_node(state: ResumeState) -> ResumeState:
    """Compute RAGAS-style metrics and persist to SQLite (v3 eval.py unchanged)."""
    try:
        jd_text = state.get("jd_text", "")
        jd_vec = embed_texts([jd_text])[0]

        resume_chunks = state.get("resume_chunks") or []
        kb_chunks_dict = state.get("kb_chunks") or {}
        kb_chunks_flat = [c for chunks in kb_chunks_dict.values() for c in chunks]

        llm_response = {
            "score": state.get("match_score", 0),
            "missing_skills": [
                m["skill"] if isinstance(m, dict) else m
                for m in (state.get("missing_skills") or [])
            ],
            "suggestions": state.get("improvement_tips") or [],
            "rewritten_bullets": [
                b.get("rewritten", "") if isinstance(b, dict) else b
                for b in (state.get("rewritten_bullets") or [])
            ],
        }

        latency_ms = state.get("latency_ms") or 0.0
        record = compute_and_log(
            job_text=jd_text,
            job_vec=jd_vec,
            resume_chunks=resume_chunks,
            kb_chunks=kb_chunks_flat,
            llm_response=llm_response,
            latency_ms=latency_ms,
        )

        return {
            **state,
            "eval_metadata": record.to_dict(),
            "messages": [
                f"[eval] ctx_rel={record.context_relevance:.3f} "
                f"faithful={record.faithfulness:.3f} "
                f"ans_rel={record.answer_relevance:.3f}"
            ],
        }
    except Exception as e:
        return {**state, "messages": [f"[eval] skipped: {e}"]}


# ── Conditional: skip rewriter if score too low ───────────────────────────────

def route_after_gap(state: ResumeState) -> str:
    if (state.get("match_score") or 0) >= 30:
        return "rewrite"
    return "skip_rewrite"


# ── Build & compile graph ─────────────────────────────────────────────────────

def build_graph():
    g = StateGraph(ResumeState)

    g.add_node("scrape_jd",    scrape_node)
    g.add_node("retrieval",    retrieval_node)
    g.add_node("parser",       parser_agent)
    g.add_node("jd_analyst",   jd_analyst_agent)
    g.add_node("gap_analyst",  gap_analyst_agent)
    g.add_node("rewriter",     rewriter_agent)
    g.add_node("eval",         eval_node)

    g.set_entry_point("scrape_jd")
    g.add_edge("scrape_jd",  "retrieval")
    g.add_edge("retrieval",  "parser")
    g.add_edge("parser",     "jd_analyst")
    g.add_edge("jd_analyst", "gap_analyst")

    g.add_conditional_edges(
        "gap_analyst",
        route_after_gap,
        {"rewrite": "rewriter", "skip_rewrite": "eval"},
    )

    g.add_edge("rewriter", "eval")
    g.add_edge("eval",     END)

    return g.compile()


agent_graph = build_graph()