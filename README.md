# Document Intelligence System — LLM RAG

**Intelligent Resume-Job Matching with Real RAG Architecture + RAGAS-style Eval**

![Python](https://img.shields.io/badge/python-3.12+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-green.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Version](https://img.shields.io/badge/version-3.0.0-red.svg)

A production-grade AI system that semantically matches resumes to job descriptions using a true multi-corpus RAG pipeline, vector embeddings, and large language models. Provides quantitative match scoring, skills-gap analysis, ATS-optimized bullet rewrites, and RAGAS-style retrieval quality metrics — all grounded in a persistent 4-corpus knowledge base.

---

## Table of Contents

- [What's New in v3](#whats-new-in-v3)
- [System Architecture](#system-architecture)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [API Reference](#api-reference)
- [Evaluation Framework](#evaluation-framework)
- [Performance](#performance)
- [Configuration](#configuration)
- [Contributing](#contributing)
- [License](#license)

---

## What's New in v3

Version 3 addresses the core architectural limitation of v2 (single-document retrieval) by introducing a persistent external knowledge base and a RAGAS-style evaluation framework.

| Dimension | v2 (Before) | v3 (Now) |
|-----------|-------------|----------|
| Retrieval corpus | Resume only (single doc) | Resume + 4 persistent KB corpora |
| Knowledge base | None | Job market data, resume examples, ATS keywords, hiring rubrics |
| RAG definition | Context selection only | True external knowledge augmentation |
| Eval framework | None | RAGAS-style: context relevance, faithfulness, answer relevance |
| Eval endpoints | None | `/eval/metrics`, `/eval/history`, `/eval/corpus/stats` |
| Prompt grounding | Resume + JD only | Resume + JD + 4 KB contexts |
| Corpus persistence | N/A | FAISS indexes built once, loaded from disk on restart |

---

## System Architecture

### RAG Pipeline

Each analysis performs dual retrieval before calling the LLM:

```
┌─────────────────┐
│   Streamlit UI  │
└────────┬────────┘
         │ HTTP/REST
         ▼
┌─────────────────┐
│     FastAPI     │
└────────┬────────┘
         │
    ┌────┴──────────────────────────────────┐
    ▼                                       ▼
┌──────────────────────┐     ┌─────────────────────────────┐
│   Resume Index       │     │   Knowledge Base (4 corpora) │
│   (per-request FAISS)│     │   job_market                 │
│                      │     │   resume_examples            │
│   PDF → chunks →     │     │   ats_keywords               │
│   embeddings → index │     │   hiring_criteria            │
└──────────┬───────────┘     └──────────────┬──────────────┘
           │                                │
           └──────────────┬─────────────────┘
                          ▼
              ┌───────────────────────┐
              │  Merged Context Prompt│
              │  + RAGAS Eval Metrics │
              └───────────┬───────────┘
                          ▼
              ┌───────────────────────┐
              │  Groq API             │
              │  Llama-3.3-70B        │
              └───────────────────────┘
```

### Knowledge Base Corpora

| Corpus | Contents | Purpose |
|--------|----------|---------|
| `job_market` | Role descriptions, responsibilities, expectations by domain | Calibrate score against real market standards |
| `resume_examples` | High-scoring bullet examples by domain and seniority | Ground rewritten bullets in proven patterns |
| `ats_keywords` | Keyword lists per role and tech stack | Surface missing ATS-critical terms |
| `hiring_criteria` | Rubrics, level definitions, ATS scoring logic, interview signals | Calibrate scoring against how humans evaluate resumes |

### Workflow

1. **PDF Parsing** — resume extracted with PyMuPDF via async ThreadPoolExecutor
2. **Chunking** — 200-word segments with 30-word overlap
3. **Embedding** — 384-dim normalized vectors via all-MiniLM-L6-v2
4. **Dual Retrieval** — top-3 resume chunks + top-2 per KB corpus (8 KB chunks total)
5. **Prompt Assembly** — resume context + JD + 4 KB contexts merged into structured prompt
6. **LLM Call** — Groq API (Llama-3.3-70B) returns structured JSON
7. **Eval Scoring** — 3 RAGAS-style metrics computed and logged per inference
8. **Response** — score, gaps, suggestions, bullets + eval metadata returned

---

## Technology Stack

### Backend
- **FastAPI** — async web framework with lifespan corpus loading
- **Python 3.12** — runtime
- **PyMuPDF (fitz)** — PDF text extraction
- **Sentence Transformers** (`all-MiniLM-L6-v2`) — 384-dim normalized embeddings
- **FAISS** — `IndexFlatL2` for resume index, `IndexFlatIP` for KB corpora (cosine on normalized vecs)
- **Groq SDK** — Llama-3.3-70B-versatile, JSON mode, temperature 0.3
- **Pydantic** — request/response validation including `EvalMetadata` schema

### Frontend
- **Streamlit** — UI with eval dashboard tab, corpus status sidebar, export options

### Infrastructure
- **Docker** — `python:3.12-slim`, EXPOSE 8080
- **Docker Compose** — backend service with `GROQ_API_KEY` env injection
- **Uvicorn** — ASGI server
- **Amazon SageMaker** — optional inference endpoint (`sagemaker/inference.py`)

---

## Project Structure

```
fadnc-document-intelligence-system--llm-rag/
├── backend/
│   ├── app.py                  # FastAPI app + lifespan corpus loading
│   ├── config.py               # Env config + KB paths
│   ├── core/
│   │   ├── logging.py          # Structured logging
│   │   └── metrics.py          # Request/latency counters
│   ├── models/
│   │   ├── prompts.py          # Dual-context prompt template
│   │   └── schemas.py          # AnalyzeResponse + EvalMetadata
│   ├── routes/
│   │   ├── analyze.py          # POST /analyze
│   │   ├── health.py           # GET /health, /readiness
│   │   └── eval.py             # GET /eval/metrics|history|corpus/stats
│   └── services/
│       ├── corpus_builder.py   # Seed + manage 4 KB FAISS indexes
│       ├── eval.py             # RAGAS-style metrics engine
│       ├── pipeline.py         # Main workflow — dual retrieval + eval
│       ├── retriever.py        # Dual search: resume index + KB
│       ├── chunker.py          # 200-word chunks, 30-word overlap
│       ├── embeddings.py       # Lazy-loaded SentenceTransformer
│       ├── llm.py              # Groq API dispatcher
│       └── parser.py           # Async PDF extraction
├── knowledge_base/             # Seed JSONs — place at project root
│   ├── job_market.json
│   ├── resume_examples.json
│   ├── ats_keywords.json
│   └── hiring_criteria.json
├── embeddings_store/
│   ├── faiss_index             # Legacy resume index
│   ├── meta.npy
│   └── knowledge_base/         # Persistent KB FAISS indexes (auto-built on first run)
├── frontend/
│   └── streamlit_app.py        # UI with eval dashboard tab
├── sagemaker/
│   └── inference.py            # SageMaker endpoint (Mistral-7B)
├── Dockerfile
├── compose.yaml
├── requirements.txt
└── .env                        # GROQ_API_KEY goes here
```

---

## Installation

### Prerequisites

- Python 3.12+
- Groq API key — free tier at [console.groq.com](https://console.groq.com)
- Docker (optional)
- Minimum 4GB RAM

### Local Setup

```bash
# 1. Clone
git clone https://github.com/fadnc/intelligent-resume-job-matching-assistant-llm-rag-sagemaker.git
cd intelligent-resume-job-matching-assistant-llm-rag-sagemaker

# 2. Virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
echo "GROQ_API_KEY=gsk_your_key_here" > .env
```

### Docker Setup

```bash
# Build and run
docker build -t resume-analyzer .
docker run -p 8080:8080 -e GROQ_API_KEY=your_key resume-analyzer

# Or with Docker Compose
GROQ_API_KEY=your_key docker compose up
```

---

## Usage

### Running Locally

**Terminal 1 — Backend:**
```bash
uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload
```

On first run, corpora are auto-built from `knowledge_base/*.json`:
```
[startup] Loading knowledge base corpora...
[corpus_builder] Building corpus: job_market
[corpus_builder] Building corpus: resume_examples
[corpus_builder] Building corpus: ats_keywords
[corpus_builder] Building corpus: hiring_criteria
[startup] Knowledge base ready.
```
Subsequent starts load indexes from disk instantly.

**Terminal 2 — Frontend:**
```bash
streamlit run frontend/streamlit_app.py
```
Opens at `http://localhost:8501`.

### Example API Request

```bash
curl -X POST http://localhost:8000/analyze \
  -F "resume=@path/to/resume.pdf" \
  -F "job_description=Software Engineer at Acme Corp..."
```

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/analyze` | Analyze resume vs job description |
| `GET` | `/health` | Liveness check |
| `GET` | `/readiness` | Readiness — verifies embedding model loaded |
| `GET` | `/metrics` | Request counts and average latency |
| `GET` | `/eval/metrics` | Aggregate RAGAS-style metrics across all inferences |
| `GET` | `/eval/history` | Per-inference eval log (`?limit=50`, max 200) |
| `GET` | `/eval/corpus/stats` | Loaded corpus doc counts and index types |
| `GET` | `/docs` | Swagger UI |
| `GET` | `/redoc` | ReDoc |

### Response Schema — `POST /analyze`

```json
{
  "score": 78,
  "missing_skills": ["Kubernetes", "GraphQL", "CI/CD pipelines"],
  "suggestions": ["Add quantified metrics...", "Include cloud tech..."],
  "rewritten_bullets": ["Architected...", "Led...", "Implemented..."],
  "eval": {
    "context_relevance": 0.812,
    "faithfulness": 0.347,
    "answer_relevance": 0.763,
    "latency_ms": 4821.3,
    "resume_chunks_used": 3,
    "kb_chunks_used": 8
  }
}
```

---

## Evaluation Framework

Three RAGAS-style metrics are computed and logged on every inference.

| Metric | What it measures | Method |
|--------|-----------------|--------|
| **Context Relevance** | Are retrieved chunks relevant to the job description? | Mean cosine similarity: JD embedding vs all retrieved chunks |
| **Faithfulness** | Does LLM output stay grounded in retrieved context? | Jaccard token overlap: context vs flattened LLM response |
| **Answer Relevance** | Is the LLM response relevant to the job description? | Cosine similarity: JD embedding vs response embedding |

Metrics are logged in-memory (capped at 200 records) and exposed via:

```bash
# Aggregate stats
GET /eval/metrics

# Per-inference history
GET /eval/history?limit=50

# Corpus health
GET /eval/corpus/stats
```

The Streamlit UI includes a dedicated **📈 Eval Metrics** tab showing per-inference scores and aggregate trends.

---

## Performance

| Operation | Duration | Notes |
|-----------|----------|-------|
| PDF Parsing | 0.5–2s | Async ThreadPoolExecutor |
| Embedding Generation | 0.3–1s | 200-word chunks, batch_size=64 |
| FAISS Resume Search | <0.1s | Top-3, IndexFlatL2 |
| FAISS KB Search | <0.1s | Top-2 per corpus × 4, IndexFlatIP |
| LLM Generation | 2–5s | Groq API, JSON mode |
| Eval Metric Computation | <0.2s | Cosine + Jaccard, in-process |
| **Total Analysis** | **3–8s** | End-to-end |

### Optimization Notes

- **Float32 precision** — reduces embedding memory vs float64
- **LRU cache** (`maxsize=100`) — caches JD embeddings, avoids re-embedding repeated queries
- **Lazy model loading** — SentenceTransformer loaded once on first request via singleton
- **KB index persistence** — corpora built once from JSON, reloaded from disk on restart
- **Context budgets** — 1500 chars resume, 1000 chars JD, 600 chars per KB corpus

---

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `GROQ_API_KEY` | — | **Required.** Groq API key from console.groq.com |
| `EMBED_MODEL` | `all-MiniLM-L6-v2` | Sentence transformer model |
| `USE_SAGEMAKER` | `False` | Route LLM calls to SageMaker endpoint |
| `SAGEMAKER_ENDPOINT` | `""` | SageMaker endpoint name if `USE_SAGEMAKER=True` |
| `BACKEND_URL` | `http://localhost:8000` | Frontend → backend URL (Streamlit env var) |

---
Deployment-ready for AWS ECS Fargate (see deploy.sh). Currently hosted on Render free tier.
---

## Contributing

Contributions are welcome. Areas of interest:

- Expanding knowledge base corpora with more domains and seniority levels
- Replacing Jaccard faithfulness with NLI-based entailment scoring
- Adding ground-truth eval datasets for offline benchmark runs
- Multi-language resume support
- PDF export of analysis results
- Unit and integration test coverage

Standard workflow: fork → feature branch → commit → pull request.

---

## License

Licensed under the MIT License. See [LICENSE](LICENSE) for details.

**Author:** Fadhil Muhammed N C  
**Repository:** [github.com/fadnc/intelligent-resume-job-matching-assistant-llm-rag-sagemaker](https://github.com/fadnc/intelligent-resume-job-matching-assistant-llm-rag-sagemaker)  
**Version:** 3.0.0 · March 2026 · Production Ready