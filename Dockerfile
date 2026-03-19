# ── Stage 1: Build knowledge base indexes ─────────────────────────────────────
FROM python:3.12-slim AS kb-builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Pin sentence-transformers to avoid triton/bitsandbytes pull on Windows
# (also keeps image lean on Render)
RUN pip install --no-cache-dir \
    "sentence-transformers==2.7.0" \
    "transformers==4.41.0" \
    "faiss-cpu" \
    "numpy" \
    "python-dotenv"

COPY knowledge_base/ ./knowledge_base/
COPY backend/ ./backend/
COPY init_kb.py .
RUN python init_kb.py

# ── Stage 2: Lean production image ────────────────────────────────────────────
FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ ./backend/
COPY knowledge_base/ ./knowledge_base/

# Baked-in KB indexes from builder — no runtime disk writes needed
COPY --from=kb-builder /build/embeddings_store/ ./embeddings_store/

EXPOSE 8000

RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]