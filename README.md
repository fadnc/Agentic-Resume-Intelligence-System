# Document Intelligence System v7
### Intelligent Resume–Job Matching · LangGraph Agents · Multi-Corpus RAG · AWS ECS

![Python](https://img.shields.io/badge/python-3.12+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green.svg)
![LangGraph](https://img.shields.io/badge/LangGraph-0.2.28-purple.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Version](https://img.shields.io/badge/version-7.0.0-red.svg)

A production-grade AI system that semantically matches resumes to job descriptions using a **multi-agent LangGraph pipeline**, persistent 4-corpus RAG knowledge base, and RAGAS-style retrieval evaluation. Deployed on AWS ECS Fargate with CI/CD via GitHub Actions and Cloudflare Tunnel for stable public access.

**Live API:** `https://eugene-coast-patients-collections.trycloudflare.com/docs`

---

## What's New in v7

| Dimension | v3 (Before) | v7 (Now) |
|-----------|-------------|----------|
| Orchestration | Single `pipeline.py` call | LangGraph `StateGraph` — 5 nodes, conditional routing |
| Agents | None | 4 specialized agents: parser, jd_analyst, gap_analyst, rewriter |
| JD input | Raw text only | Raw text **or** URL (LinkedIn / Naukri scraper) |
| Gap analysis | Cosine similarity only | Embeddings + LLM reasoning grounded in hiring_criteria KB |
| Rewriter | Generic suggestions | Bullet-level rewrites grounded in resume_examples KB |
| Response | score + missing_skills | + matched_skills, gap_summary, tailored_summary, quick_wins, agent_trace |
| Deployment | Render free tier | AWS ECS Fargate — 4 independent services |
| Observability | In-memory metrics | MLflow experiment tracking + Grafana dashboards |
| CI/CD | None | GitHub Actions — test → Docker build → ECR push → ECS deploy |
| Public access | IP-based | Cloudflare Tunnel — stable HTTPS URL, no load balancer cost |

---

## Architecture

### Agent Graph (LangGraph StateGraph)

```
Input (PDF resume + JD text or URL)
          │
          ▼
   scrape_jd node ──── optional: Playwright/BS4 scraper
          │
          ▼
   retrieval node ──── dual FAISS search:
          │              top-3 resume chunks + top-2 per KB corpus (8 KB chunks)
          ▼
   parser agent ─────── extracts skills, experience, education from resume
          │              grounded in retrieved resume chunks
          ▼
   jd_analyst agent ─── ranks must-have vs nice-to-have requirements
          │              grounded in KB: job_market + ats_keywords
          ▼
   gap_analyst agent ── cosine similarity + LLM scoring (0–100)
          │              grounded in KB: hiring_criteria
          │
     score ≥ 30? ──── YES ──► rewriter agent ── tailored bullets + summary
          │                                       grounded in KB: resume_examples
          │ NO
          ▼
       eval node ─────── RAGAS-style metrics logged to SQLite
          │
          ▼
        END → JSON response
```

### Knowledge Base (4 Persistent FAISS Corpora)

| Corpus | Contents | Used by |
|--------|----------|---------|
| `job_market` | Role descriptions, responsibilities by domain | jd_analyst |
| `resume_examples` | High-scoring bullets by domain + seniority | rewriter |
| `ats_keywords` | Keyword lists per role and tech stack | jd_analyst |
| `hiring_criteria` | Rubrics, ATS scoring logic, interview signals | gap_analyst |

Indexes are built once at Docker image build time (`RUN python init_kb.py`) and loaded from disk on every startup — no cold-build latency in production.

---

## Technology Stack

### ML & Backend
- **LangGraph** — agent orchestration with `StateGraph`, shared `ResumeState` TypedDict, conditional routing
- **FastAPI** — async API with lifespan KB loading, PDF upload, background MLflow logging
- **Groq API** — `llama-3.3-70b-versatile`, JSON mode, temperature 0.1 for structured outputs
- **Sentence Transformers** (`all-MiniLM-L6-v2`) — 384-dim normalized embeddings, LRU-cached
- **FAISS** — `IndexFlatIP` (cosine on normalized vectors) for KB corpora, `IndexFlatL2` for per-request resume index
- **PyMuPDF** — async PDF extraction via `ThreadPoolExecutor`
- **Playwright + BeautifulSoup** — JD scraping from LinkedIn, Naukri, and generic URLs
- **MLflow** — experiment tracking: match score, latency, gap count per inference run
- **SQLite** — persistent RAGAS-style eval log (context relevance, faithfulness, answer relevance)

### DevOps & Infrastructure
- **Docker** — multi-stage build, KB indexes baked in at build time, non-root user
- **AWS ECS Fargate** — 4 independent services (backend, frontend, MLflow, Grafana)
- **AWS ECR** — container registry
- **AWS Secrets Manager** — GROQ_API_KEY injection at runtime
- **GitHub Actions** — CI/CD: pytest → docker build → ECR push → ECS rolling deploy
- **Cloudflare Tunnel** — stable public HTTPS without a load balancer
- **Prometheus + Grafana** — request rate, latency, match score distribution

---

## Project Structure

```
├── backend/
│   ├── agents/
│   │   ├── state.py              # ResumeState TypedDict — shared across all agents
│   │   ├── graph.py              # LangGraph StateGraph — replaces pipeline.py
│   │   ├── parser_agent.py       # Extracts skills/exp/edu from resume
│   │   ├── jd_analyst_agent.py   # Ranks JD requirements, grounded in KB
│   │   ├── gap_analyst_agent.py  # Cosine sim + LLM scoring, 0–100
│   │   └── rewriter_agent.py     # Bullet rewrites + tailored summary
│   ├── tools/
│   │   ├── retrieval.py          # Wraps v3 dual-retrieval for graph node
│   │   └── scraper.py            # LinkedIn/Naukri/generic JD scraper
│   ├── core/
│   │   ├── logging.py            # Structured logging
│   │   └── metrics.py            # Request/latency counters
│   ├── models/
│   │   ├── schemas.py            # AnalyzeResponse + EvalMetadata (v7 extended)
│   │   └── prompts.py            # Legacy prompt template (reference)
│   ├── routes/
│   │   ├── analyze.py            # POST /analyze — invokes agent_graph
│   │   ├── health.py             # GET /health, /readiness
│   │   └── eval.py               # GET /eval/metrics|history|corpus/stats
│   └── services/
│       ├── corpus_builder.py     # 4-corpus FAISS manager, disk persistence
│       ├── retriever.py          # Dual search: resume index + KB corpora
│       ├── eval.py               # RAGAS metrics + SQLite persistence
│       ├── embeddings.py         # Lazy-loaded SentenceTransformer singleton
│       ├── chunker.py            # 300-word chunks, 50-word overlap
│       ├── parser.py             # Async PDF extraction via PyMuPDF
│       └── llm.py                # Groq dispatcher + SageMaker stub
├── knowledge_base/               # Seed JSONs for 4 KB corpora
├── flask_frontend/               # Flask UI proxying to backend
├── sagemaker/                    # Mistral-7B SageMaker endpoint stub
├── aws/
│   ├── task-definitions/         # ECS task definitions per service
│   ├── deploy-all-services.sh    # One-command ECS deploy
│   └── get-urls-all.sh           # Fetch live IPs for all services
├── .github/workflows/ci-cd.yml   # GitHub Actions pipeline
├── docker/
│   ├── prometheus.yml            # Prometheus scrape config
│   └── grafana/                  # Grafana datasource provisioning
├── Dockerfile                    # Multi-stage, KB baked in at build time
├── Dockerfile.frontend
├── compose.yaml                  # Local dev: all 6 services
├── init_kb.py                    # Seeds FAISS indexes from JSON
└── requirements.txt
```

---

## Local Setup

```bash
git clone https://github.com/fadnc/intelligent-resume-job-matching-assistant-llm-rag-sagemaker.git
cd intelligent-resume-job-matching-assistant-llm-rag-sagemaker

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
playwright install chromium     # for JD URL scraping

cp .env.example .env
# Add GROQ_API_KEY to .env

python init_kb.py               # Build FAISS indexes once
uvicorn backend.app:app --reload --port 8000
```

### Docker (all services)

```bash
GROQ_API_KEY=your_key docker compose up --build
```

| Service | URL |
|---------|-----|
| API | http://localhost:8000 |
| API Docs | http://localhost:8000/docs |
| Frontend | http://localhost:5000 |
| MLflow | http://localhost:5001 |
| Grafana | http://localhost:3000 |

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/analyze` | Full agent pipeline — PDF + JD text or URL |
| `GET` | `/health` | Liveness check |
| `GET` | `/readiness` | Verifies embedding model loaded |
| `GET` | `/metrics` | Request counts and average latency |
| `GET` | `/eval/metrics` | Aggregate RAGAS-style metrics |
| `GET` | `/eval/history` | Per-inference eval log |
| `GET` | `/eval/corpus/stats` | KB corpus doc counts and index types |
| `GET` | `/docs` | Swagger UI |

### Response Schema

```json
{
  "score": 74,
  "matched_skills": ["Python", "PyTorch", "FastAPI", "FAISS"],
  "missing_skills": [
    { "skill": "AWS", "importance": "critical", "learnable_in_weeks": 4 }
  ],
  "gap_summary": "Strong ML fit. Main gap is cloud deployment experience.",
  "rewritten_bullets": [
    {
      "section": "ML Research at Digital University Kerala",
      "original": "Built models using MIMIC-III data",
      "rewritten": "Developed and deployed production ML pipelines on MIMIC-III (61K+ ICU stays), achieving AUROC > 0.85 across 6 clinical outcomes",
      "reason": "JD emphasises production ML pipelines and quantified impact"
    }
  ],
  "tailored_summary": "ML Engineer with experience building...",
  "improvement_tips": ["Get AWS Cloud Practitioner cert — directly addresses the biggest gap"],
  "quick_wins": ["Add 'deployed to cloud' to existing project bullets"],
  "agent_trace": [
    "[scraper] JD fetched from URL",
    "[retrieval] 3 resume chunks, 8 KB chunks",
    "[parser] 12 skills, 2 roles",
    "[jd_analyst] 5 must-have, 4 nice-to-have, domain=ML Engineering",
    "[gap_analyst] score=74/100 gaps=1 semantic=0.87",
    "[rewriter] 3 bullets rewritten, tailored summary generated",
    "[eval] ctx_rel=0.812 faithful=0.347 ans_rel=0.763"
  ],
  "eval": {
    "context_relevance": 0.812,
    "faithfulness": 0.347,
    "answer_relevance": 0.763,
    "latency_ms": 4821.3,
    "resume_chunks_used": 3,
    "kb_chunks_used": 8
  },
  "processing_time_ms": 4230.5
}
```

---

## Deployment

### AWS ECS Fargate

4 independent Fargate services, each with its own task definition:

```bash
bash aws/deploy-all-services.sh
```

CI/CD via GitHub Actions on every push to `main`:
- pytest with coverage
- Docker build + push to ECR
- ECS rolling deploy with task definition patching

### Demo (start/stop)

```bash
# Start (2 min to be live)
bash start-for-demo.sh

# Stop (billing pauses immediately)
bash stop-after-demo.sh
```

---

## Evaluation Framework

Three RAGAS-style metrics computed and persisted to SQLite on every inference:

| Metric | Measures | Method |
|--------|----------|--------|
| **Context Relevance** | Retrieved chunks vs job description | Mean cosine similarity |
| **Faithfulness** | LLM output grounded in context | Jaccard token overlap |
| **Answer Relevance** | Response relevant to job description | Cosine similarity |

---

## Performance

| Operation | Duration |
|-----------|----------|
| PDF parsing | 0.5–2s |
| Embedding + FAISS search | 0.3–1s |
| LLM generation (4 agent calls) | 6–12s |
| RAGAS eval computation | <0.2s |
| **Total end-to-end** | **7–15s** |

---

## Configuration

| Variable | Description |
|----------|-------------|
| `GROQ_API_KEY` | Required. Get free key at console.groq.com |
| `MLFLOW_TRACKING_URI` | MLflow server URL (default: http://localhost:5001) |
| `USE_SAGEMAKER` | Route LLM to SageMaker endpoint (default: false) |
| `SAGEMAKER_ENDPOINT` | SageMaker endpoint name |

---

## License

MIT License — see [LICENSE](LICENSE) for details.

**Author:** Fadhil Muhammed N C · [linkedin.com/in/fadhilmuhd](https://linkedin.com/in/fadhilmuhd) · [github.com/fadnc](https://github.com/fadnc)

**Version:** 7.0.0 · July 2026