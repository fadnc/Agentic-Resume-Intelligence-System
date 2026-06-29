"""
backend/app.py — v7
FastAPI app with LangGraph agent graph replacing pipeline.py.
All v3 lifespan KB loading, middleware, and routes preserved.
"""
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from backend.routes.analyze import router as analyze_router
from backend.routes.health import router as health_router
from backend.routes.eval import router as eval_router
from backend.core.logging import configure_logging
from backend.core.metrics import metrics
from backend.services.corpus_builder import load_all_corpora

logger = configure_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("[startup] Loading knowledge base corpora...")
    try:
        load_all_corpora()
        logger.info("[startup] Knowledge base ready.")
    except Exception as e:
        logger.error(f"[startup] Failed to load corpora: {e}")
    # Pre-import graph so LangGraph compiles on startup, not first request
    try:
        from backend.agents.graph import agent_graph  # noqa: F401
        logger.info("[startup] LangGraph agent graph compiled.")
    except Exception as e:
        logger.error(f"[startup] Graph compile error: {e}")
    yield
    logger.info("[shutdown] Shutting down.")


app = FastAPI(
    title="Document Intelligence System — v7",
    description="Multi-agent resume analysis: LangGraph agents + 4-corpus RAG + RAGAS eval",
    version="7.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_metrics(request: Request, call_next):
    start = time.time()
    try:
        response = await call_next(request)
        latency = time.time() - start
        metrics.record(latency)
        logger.info(f"{request.method} {request.url.path} — {latency:.3f}s")
        return response
    except Exception as e:
        metrics.error()
        logger.error(f"Error: {e}")
        raise


@app.get("/")
def root():
    return {"message": "Document Intelligence System v7 — LangGraph + RAG"}


@app.get("/metrics")
def get_metrics():
    return {
        "total_requests": metrics.total_requests,
        "total_errors":   metrics.total_errors,
        "average_latency": metrics.avg_latency,
    }


app.include_router(analyze_router)
app.include_router(health_router)
app.include_router(eval_router)