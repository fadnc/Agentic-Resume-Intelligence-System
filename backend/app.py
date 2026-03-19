import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from backend.routes.analyze import router as analyze_router
from backend.routes.health import router as health_router
from backend.routes.eval import router as eval_router
from backend.core.logging import configure_logging
from backend.core.metrics import metrics
from backend.services.corpus_builder import load_all_corpora

logger = configure_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load knowledge base corpora on startup."""
    logger.info("[startup] Loading knowledge base corpora...")
    try:
        load_all_corpora()
        logger.info("[startup] Knowledge base ready.")
    except Exception as e:
        logger.error(f"[startup] Failed to load corpora: {e}")
    yield
    logger.info("[shutdown] Shutting down.")


app = FastAPI(title="Resume AI — Document Intelligence System", lifespan=lifespan)


@app.middleware("http")
async def add_metrics(request: Request, call_next):
    start = time.time()
    try:
        response = await call_next(request)
        latency = time.time() - start
        metrics.record(latency)
        logger.info(f"{request.method} {request.url.path} - {latency:.4f}s")
        return response
    except Exception as e:
        metrics.error()
        logger.error(f"Error: {str(e)}")
        raise e


@app.get("/")
def root():
    return {"message": "Resume AI — Document Intelligence System v3"}


@app.get("/metrics")
def get_metrics():
    return {
        "total_requests": metrics.total_requests,
        "total_errors": metrics.total_errors,
        "average_latency": metrics.avg_latency,
    }


app.include_router(analyze_router)
app.include_router(health_router)
app.include_router(eval_router)