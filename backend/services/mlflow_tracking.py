"""MLflow logging for resume analysis requests."""

import logging
import sys
from typing import Any, Optional

import mlflow

from backend.config import MLFLOW_EXPERIMENT, MLFLOW_TRACKING_URI

logger = logging.getLogger("resume-ai")

# Some MLflow versions print the run link with Unicode symbols when a run
# finishes. Use a UTF-8-capable stream so that message cannot interrupt run
# finalization on Windows consoles using a legacy code page.
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (OSError, ValueError):
        pass


def log_analysis_run(
    latency_ms: float,
    *,
    jd_url_provided: bool,
    status: str,
    match_score: Optional[float] = None,
    missing_skill_count: int = 0,
    matched_skill_count: int = 0,
    resume_chunk_count: int = 0,
    kb_chunk_count: int = 0,
    eval_metrics: Optional[dict[str, Any]] = None,
) -> None:
    """Log aggregate analysis results without sending resume or JD text."""
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT)

        with mlflow.start_run(run_name="resume-analysis"):
            mlflow.set_tags({
                "analysis.status": status,
                "analysis.jd_url_provided": str(jd_url_provided).lower(),
                "analysis.workflow": "langgraph-rag",
            })
            mlflow.log_params({
                "jd_url_provided": jd_url_provided,
                "resume_chunk_count": resume_chunk_count,
                "kb_chunk_count": kb_chunk_count,
            })

            metrics = {
                "latency_ms": float(latency_ms),
                "pipeline_success": 1.0 if status == "success" else 0.0,
                "missing_skill_count": float(missing_skill_count),
                "matched_skill_count": float(matched_skill_count),
            }
            if match_score is not None:
                metrics["match_score"] = float(match_score)

            eval_metrics = eval_metrics or {}
            for name in ("context_relevance", "faithfulness", "answer_relevance"):
                value = eval_metrics.get(name)
                if value is not None:
                    metrics[name] = float(value)

            mlflow.log_metrics(metrics)
    except Exception as exc:
        # Tracking is best effort: an MLflow outage must not fail an analysis.
        logger.warning("MLflow logging failed (%s)", type(exc).__name__)
