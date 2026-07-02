"""
backend/routes/analyze.py — v7
POST /analyze — runs the full LangGraph agent pipeline.
Accepts PDF upload + job_description form field (same as v3).
Also accepts optional jd_url for scraping.
"""
import time
import io
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks
from typing import Optional

from backend.agents.graph import agent_graph
from backend.services.parser import extract_text_from_pdf
from backend.models.schemas import AnalyzeResponse

router = APIRouter()


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze(
    resume: UploadFile = File(...),
    job_description: str = Form(...),
    jd_url: Optional[str] = Form(default=None),
):
    """
    Full agent pipeline:
      PDF parse → JD scrape (optional) → retrieval → parser →
      jd_analyst → gap_analyst → rewriter → RAGAS eval
    """
    if not resume.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=422, detail="Only PDF resumes are supported.")

    file_bytes = await resume.read()
    resume_text = await extract_text_from_pdf(file_bytes)

    if not resume_text.strip():
        raise HTTPException(status_code=422, detail="Could not extract text from PDF.")

    start = time.time()

    initial_state = {
        "resume_text":      resume_text,
        "jd_text":          job_description,
        "jd_url":           jd_url,
        "resume_chunks":    None,
        "kb_chunks":        None,
        "resume_index_key": None,
        "parsed_resume":    None,
        "parsed_jd":        None,
        "match_score":      None,
        "matched_skills":   None,
        "missing_skills":   None,
        "gap_summary":      None,
        "quick_wins":       None,
        "rewritten_bullets": None,
        "tailored_summary": None,
        "improvement_tips": None,
        "eval_metadata":    None,
        "messages":         [],
        "error":            None,
        "latency_ms":       None,
    }

    result = agent_graph.invoke(initial_state)

    if result.get("error"):
        raise HTTPException(status_code=500, detail=result["error"])

    elapsed_ms = (time.time() - start) * 1000

    # v3-compatible response shape + v7 extras
    return AnalyzeResponse(
        score=result.get("match_score") or 0,
        missing_skills=[
            m["skill"] if isinstance(m, dict) else m
            for m in (result.get("missing_skills") or [])
        ],
        suggestions=result.get("improvement_tips") or [],
        rewritten_bullets=[
            b.get("rewritten", "") if isinstance(b, dict) else b
            for b in (result.get("rewritten_bullets") or [])
        ],
        # v7 extras
        matched_skills=result.get("matched_skills") or [],
        gap_summary=result.get("gap_summary") or "",
        tailored_summary=result.get("tailored_summary") or "",
        quick_wins=result.get("quick_wins") or [],
        agent_trace=result.get("messages") or [],
        eval=result.get("eval_metadata"),
        processing_time_ms=round(elapsed_ms, 1),
    )