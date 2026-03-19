from fastapi import APIRouter
from backend.services.embeddings import get_model

router = APIRouter()

@router.get("/health")
def health():
    return {"status": "ok"}

@router.get("/readiness")
def readiness():
    try:
        model = get_model()
        if model:
            return {"status": "ready"}
        return {"status": "not_ready"}
    except Exception:
        return {"status": "error"}