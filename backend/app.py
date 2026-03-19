import time
from fastapi import FastAPI, Request
from backend.routes.analyze import router as analyze_router
from backend.routes.health import router as health_router
from backend.core.logging import configure_logging
from backend.core.metrics import metrics

app = FastAPI(title="Resume LLM Assistant v3")

logger = configure_logging()

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
    return {"message": "Resume AI API Running"}

@app.get("/metrics")
def get_metrics():
    return {
        "total_requests": metrics.total_requests,
        "total_errors": metrics.total_errors,
        "average_latency": metrics.avg_latency
    }

app.include_router(analyze_router)
app.include_router(health_router)