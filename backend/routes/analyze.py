"""
backend/routes/analyze.py — v7
POST /analyze — runs the full LangGraph agent pipeline.
Accepts PDF upload + job_description form field (same as v3).
Also accepts optional jd_url for scraping.
"""
import time
import io
import logging
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks
from typing import Optional

from backend.agents.graph import agent_graph
from backend.services.parser import extract_text_from_pdf
from backend.services.mlflow_tracking import log_analysis_run
from backend.models.schemas import AnalyzeResponse
from backend.security.input_guard import (
    MAX_JOB_DESCRIPTION_CHARS,
    MAX_PDF_BYTES,
    MAX_RESUME_TEXT_CHARS,
    check_untrusted_text,
)
from backend.services.evidence import build_requirement_evidence

router = APIRouter()
logger = logging.getLogger("resume-ai")


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze(
    background_tasks: BackgroundTasks,
    resume: UploadFile = File(...),
    job_description: str = Form(...),
    jd_url: Optional[str] = Form(default=None),
):
    """
    Full agent pipeline:
      PDF parse → JD scrape (optional) → retrieval → parser →
      jd_analyst → gap_analyst → rewriter → RAGAS eval
    """
    if not resume.filename or not resume.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=422, detail="Only PDF resumes are supported.")

    file_bytes = await resume.read(MAX_PDF_BYTES + 1)
    if len(file_bytes) > MAX_PDF_BYTES:
        raise HTTPException(status_code=413, detail="Resume PDF must be 10 MB or smaller.")
    if not file_bytes.startswith(b"%PDF-"):
        raise HTTPException(status_code=422, detail="The uploaded file is not a valid PDF.")
    resume_text = await extract_text_from_pdf(file_bytes)

    if not resume_text.strip():
        raise HTTPException(status_code=422, detail="Could not extract text from PDF.")

    try:
        resume_text = check_untrusted_text(
            resume_text, label="Resume text", max_chars=MAX_RESUME_TEXT_CHARS
        )
        job_description = check_untrusted_text(
            job_description, label="Job description", max_chars=MAX_JOB_DESCRIPTION_CHARS
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

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

    try:
        result = agent_graph.invoke(initial_state)
    except Exception as exc:
        logger.exception("[analyze] agent pipeline failed")
        elapsed_ms = (time.time() - start) * 1000
        log_analysis_run(
            elapsed_ms,
            jd_url_provided=bool(jd_url),
            status="error",
        )
        if exc.__class__.__name__ == "AuthenticationError":
            raise HTTPException(
                status_code=502,
                detail="Groq rejected GROQ_API_KEY. Replace it in .env with a valid key and restart the backend.",
            ) from exc
        if exc.__class__.__name__ == "APIConnectionError":
            raise HTTPException(
                status_code=502,
                detail="Could not connect to the Groq API. Check network access from the backend and retry.",
            ) from exc
        raise

    elapsed_ms = (time.time() - start) * 1000
    kb_chunks = result.get("kb_chunks") or {}
    tracking_data = {
        "match_score": result.get("match_score"),
        "missing_skill_count": len(result.get("missing_skills") or []),
        "matched_skill_count": len(result.get("matched_skills") or []),
        "resume_chunk_count": len(result.get("resume_chunks") or []),
        "kb_chunk_count": sum(len(chunks) for chunks in kb_chunks.values()),
        "eval_metrics": result.get("eval_metadata") or {},
    }

    if result.get("error"):
        log_analysis_run(
            elapsed_ms,
            jd_url_provided=bool(jd_url),
            status="error",
            **tracking_data,
        )
        raise HTTPException(status_code=500, detail=result["error"])

    background_tasks.add_task(
        log_analysis_run,
        elapsed_ms,
        jd_url_provided=bool(jd_url),
        status="success",
        **tracking_data,
    )

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
        tailored_resume=result.get("tailored_resume") or result.get("parsed_resume") or {},
        interview_questions=result.get("interview_questions") or [],
        requirement_evidence=build_requirement_evidence(
            resume_text, result.get("parsed_jd") or {}
        ),
        job_title=(result.get("parsed_jd") or {}).get("title", ""),
        quick_wins=result.get("quick_wins") or [],
        agent_trace=result.get("messages") or [],
        eval=result.get("eval_metadata"),
        processing_time_ms=round(elapsed_ms, 1),
    )
