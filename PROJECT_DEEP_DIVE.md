# Document Intelligence System v7 — Complete Deep Dive

> A comprehensive guide to every concept, theory, and design decision in this project.

---

## Table of Contents

1. [What This Project Does](#1-what-this-project-does)
2. [Core Concepts & Theories](#2-core-concepts--theories)
   - [RAG (Retrieval-Augmented Generation)](#21-rag-retrieval-augmented-generation)
   - [Vector Embeddings](#22-vector-embeddings)
   - [FAISS (Facebook AI Similarity Search)](#23-faiss-facebook-ai-similarity-search)
   - [Cosine Similarity vs L2 Distance](#24-cosine-similarity-vs-l2-distance)
   - [Text Chunking](#25-text-chunking)
   - [LLMs and Prompt Engineering](#26-llms-and-prompt-engineering)
   - [Multi-Agent Systems (LangGraph)](#27-multi-agent-systems-langgraph)
   - [RAGAS Evaluation Framework](#28-ragas-evaluation-framework)
3. [Architecture Walkthrough](#3-architecture-walkthrough)
   - [The Agent Graph Pipeline](#31-the-agent-graph-pipeline)
   - [The Four Knowledge Base Corpora](#32-the-four-knowledge-base-corpora)
   - [Dual Retrieval Strategy](#33-dual-retrieval-strategy)
4. [Agent-by-Agent Breakdown](#4-agent-by-agent-breakdown)
5. [Services Layer (The Engine Room)](#5-services-layer-the-engine-room)
6. [API & Frontend](#6-api--frontend)
7. [DevOps & Deployment](#7-devops--deployment)
8. [Key Design Decisions & Why](#8-key-design-decisions--why)

---

## 1. What This Project Does

This system takes a **PDF resume** and a **job description** (text or URL), and produces:

- A **match score** (0–100) — how well the resume fits the job
- **Matched skills** — what the candidate already has
- **Missing skills** — gaps, ranked by importance, with estimated learning time
- **Rewritten resume bullets** — existing experience rephrased to match JD language
- **Tailored professional summary** — a custom summary targeting the specific role
- **Quick wins** — things the candidate can add to their resume *right now*
- **Quality metrics** — RAGAS-style evaluation of retrieval and generation quality

It does this using a **multi-agent pipeline** where 4 specialized AI agents collaborate, each grounded in a **persistent knowledge base** of hiring standards, ATS rules, and resume best practices.

---

## 2. Core Concepts & Theories

### 2.1 RAG (Retrieval-Augmented Generation)

**What it is:** RAG is a technique that improves LLM outputs by first *retrieving* relevant information from a knowledge base, then *injecting* that information into the LLM prompt as context.

**Why it matters here:** Without RAG, an LLM answering "how well does this resume match this job?" would rely solely on its training data — which is generic, potentially outdated, and not grounded in real hiring criteria. With RAG, the LLM's analysis is **grounded** in:
- The actual resume text (via FAISS-indexed resume chunks)
- Curated hiring rubrics, ATS keyword lists, and resume best practices (via the 4 KB corpora)

**How it works in this project:**

```
User inputs (resume + JD)
        │
        ▼
  ┌─────────────┐     ┌──────────────────┐
  │ Embed the   │────▶│ Search FAISS     │
  │ JD as a     │     │ indexes for      │
  │ query vector│     │ relevant chunks  │
  └─────────────┘     └──────────────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │ Inject retrieved │
                    │ chunks into LLM  │
                    │ prompts as       │
                    │ grounding context│
                    └──────────────────┘
                              │
                              ▼
                      Grounded LLM output
```

**Key insight:** The LLM never sees the *entire* knowledge base — only the most relevant chunks. This keeps prompts small and focused.

---

### 2.2 Vector Embeddings

**What they are:** Embeddings are dense numerical representations (vectors) of text. Texts with similar *meaning* end up close together in vector space, even if they use different words.

**The model used:** `sentence-transformers/all-MiniLM-L6-v2`
- Produces **384-dimensional** vectors
- Trained on 1B+ sentence pairs for semantic similarity
- Small, fast (~33M parameters), good quality-to-speed ratio
- Outputs are **L2-normalized** (unit vectors), which makes cosine similarity = inner product

**How it's used:**

```python
# From backend/services/embeddings.py
def embed_texts(texts):
    model = get_model()
    vectors = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True  # ← critical: enables cosine via inner product
    )
    return np.array(vectors).astype("float32")
```

**Why normalization matters:** When vectors are normalized to unit length (‖v‖ = 1), the inner product (dot product) equals cosine similarity. This lets FAISS use `IndexFlatIP` (inner product) which is computationally cheaper than computing full cosine similarity.

**Singleton pattern:** The model is loaded lazily (first call) and cached globally. This avoids loading the ~90MB model multiple times.

---

### 2.3 FAISS (Facebook AI Similarity Search)

**What it is:** FAISS is a library for efficient similarity search over dense vectors. Given a query vector, it finds the *k nearest neighbors* in a collection of vectors.

**Two index types used in this project:**

| Index | Where Used | Why |
|-------|-----------|-----|
| `IndexFlatIP` (Inner Product) | Knowledge base corpora | Vectors are normalized → IP = cosine similarity. Higher score = more similar. |
| `IndexFlatL2` (L2/Euclidean Distance) | Per-request resume index | Simpler; lower distance = more similar. Used for within-resume chunk retrieval. |

**"Flat" means:** No approximation — it compares the query against *every* vector in the index (brute force). This is fine for small datasets (the KB has ~50 documents total). For millions of vectors, you'd use approximate indexes like `IndexIVFFlat` or `IndexHNSW`.

**Persistence:** KB indexes are built once (`python init_kb.py` or at Docker build time) and saved to disk as `.faiss` files. On startup, they're loaded from disk into memory — no re-embedding needed.

---

### 2.4 Cosine Similarity vs L2 Distance

**Cosine similarity** measures the *angle* between two vectors (ignoring magnitude):

```
cos(A, B) = (A · B) / (‖A‖ × ‖B‖)
```

- Range: -1 to 1 (for normalized text embeddings, typically 0 to 1)
- 1.0 = identical direction (same meaning)
- 0.0 = orthogonal (unrelated)

**L2 (Euclidean) distance** measures the straight-line distance:

```
L2(A, B) = √(Σ(Aᵢ - Bᵢ)²)
```

- Range: 0 to ∞
- 0.0 = identical vectors
- For normalized vectors: L2² = 2 - 2×cos(A,B), so they're mathematically equivalent

**Why this project uses both:** The KB uses inner product (cosine) because we care about *semantic direction* — "is this hiring rubric about the same topic as the JD?" The resume index uses L2 because it was carried forward from v3 and works fine for the same purpose on normalized vectors.

---

### 2.5 Text Chunking

**The problem:** Embedding models have token limits and perform better on shorter, focused text. A 3-page resume as a single string would produce a diluted embedding.

**The solution:** Split text into overlapping chunks.

```python
# From backend/services/chunker.py
def chunk_text(text, size=300, overlap=50):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += size - overlap  # slide window
    return chunks
```

**Parameters:**
- `size=300` words per chunk — large enough for a meaningful passage, small enough for focused embeddings
- `overlap=50` words — ensures sentences at chunk boundaries aren't split, maintaining context continuity

**Visual:**
```
[--------- chunk 1 (300 words) ---------]
                            [--------- chunk 2 (300 words) ---------]
                                                        [--------- chunk 3 ... ]
                            ↑                           ↑
                        50-word overlap             50-word overlap
```

---

### 2.6 LLMs and Prompt Engineering

**The LLM used:** `llama-3.3-70b-versatile` via Groq API

**Why Groq?** Groq provides extremely fast inference (~10x faster than OpenAI) by running LLMs on their custom LPU (Language Processing Unit) hardware. The free tier is sufficient for development.

**Key LLM settings:**

| Parameter | Value | Why |
|-----------|-------|-----|
| `temperature` | 0.1–0.2 | Low temperature = deterministic, consistent outputs. Critical for structured JSON responses. |
| `max_tokens` | 1000–1500 | Limits response length to control costs and latency. |
| `response_format` | `{"type": "json_object"}` | Forces the LLM to output valid JSON — eliminates parsing failures. |

**Prompt engineering patterns used:**

1. **Role assignment:** Each agent prompt starts with "You are a [specific role]" — e.g., "expert technical recruiter", "career coach", "expert resume writer"
2. **Context injection:** Retrieved KB chunks are inserted into the prompt as grounding context
3. **Output schema enforcement:** Every prompt specifies the exact JSON structure expected
4. **Constraint specification:** Rules like "DO NOT fabricate experience" and "Mirror JD language exactly"
5. **Fallback parsing:** If JSON parsing fails, a regex extracts the JSON object: `re.search(r"\{.*\}", raw, re.DOTALL)`

---

### 2.7 Multi-Agent Systems (LangGraph)

**What is LangGraph?** LangGraph is a framework for building stateful, multi-step AI applications as directed graphs. Each node is a function (an "agent" or "tool"), and edges define the execution order.

**Key concepts:**

**StateGraph:** The graph structure where nodes share a common state object (`ResumeState`):

```python
class ResumeState(TypedDict):
    resume_text: str
    jd_text: str
    parsed_resume: Optional[dict]
    parsed_jd: Optional[dict]
    match_score: Optional[int]
    # ... 15+ fields total
    messages: Annotated[list, add]  # append-only trace log
```

**`Annotated[list, add]`:** This is a LangGraph reducer. The `add` operator means when a node returns `{"messages": ["new entry"]}`, it *appends* to the existing list rather than *replacing* it. This creates a running trace log across all agents.

**Conditional routing:** After the gap analyst, the graph makes a decision:

```python
def route_after_gap(state):
    if (state.get("match_score") or 0) >= 30:
        return "rewrite"       # worth rewriting
    return "skip_rewrite"      # too poor a match, skip
```

This is a **conditional edge** — the next node depends on the state. Scores below 30 skip the rewriter (no point rewriting a resume for a completely unrelated job).

**Why agents instead of a single pipeline?**
- **Separation of concerns:** Each agent has one job with its own prompt and KB context
- **Independent iteration:** You can improve the rewriter without touching the parser
- **Conditional logic:** Agents can be skipped based on state
- **Traceability:** The `messages` trace shows exactly what each agent did

---

### 2.8 RAGAS Evaluation Framework

**What is RAGAS?** RAGAS (Retrieval Augmented Generation Assessment) is a framework for evaluating RAG systems. This project implements three RAGAS-inspired metrics:

#### Metric 1: Context Relevance (0–1)

**Question:** "Were the retrieved chunks actually relevant to the job description?"

**Method:** Mean cosine similarity between the JD embedding and each retrieved chunk's embedding.

```python
def _mean_cosine(query_vec, chunks):
    chunk_vecs = embed_texts(chunks)
    sims = [cosine_similarity(query_vec, cv) for cv in chunk_vecs]
    return np.mean(sims)
```

**Interpretation:** 0.8+ = highly relevant retrieval. Below 0.5 = retrieval is pulling irrelevant chunks.

#### Metric 2: Faithfulness (0–1)

**Question:** "Is the LLM's output grounded in the retrieved context, or is it hallucinating?"

**Method:** Jaccard token overlap between the context text and the response text.

```python
def _jaccard_token_overlap(text_a, text_b):
    tokens_a = set(text_a.lower().split())
    tokens_b = set(text_b.lower().split())
    return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)
```

**Interpretation:** Higher = more words in the response appear in the context = more grounded. This is a **proxy metric** — the code notes it should be replaced with NLI (Natural Language Inference) entailment for production accuracy.

#### Metric 3: Answer Relevance (0–1)

**Question:** "Does the LLM's response actually address the job description?"

**Method:** Cosine similarity between the JD embedding and the response embedding.

**Persistence:** All three metrics are logged to SQLite on every inference, creating a persistent evaluation history for trend analysis.

---

## 3. Architecture Walkthrough

### 3.1 The Agent Graph Pipeline

```
POST /analyze (PDF + JD text/URL)
        │
        ▼
   [PDF Parser]  ← PyMuPDF extracts text from PDF
        │
        ▼
   ┌─────────────────────────────────────────────────┐
   │              LangGraph StateGraph                │
   │                                                  │
   │  1. scrape_jd ─── if URL provided, fetch JD     │
   │        │                                         │
   │        ▼                                         │
   │  2. retrieval ─── build resume FAISS index       │
   │        │          + dual search (resume + KB)    │
   │        ▼                                         │
   │  3. parser ────── extract skills/exp/edu         │
   │        │          (LLM + resume chunks)          │
   │        ▼                                         │
   │  4. jd_analyst ── rank must-have vs nice-to-have │
   │        │          (LLM + job_market + ats_kw KB) │
   │        ▼                                         │
   │  5. gap_analyst ─ cosine sim + LLM scoring       │
   │        │          (LLM + hiring_criteria KB)     │
   │        │                                         │
   │   score >= 30?                                   │
   │     YES ──► 6. rewriter ── bullet rewrites       │
   │     │           (LLM + resume_examples KB)       │
   │     │                                            │
   │     ▼                                            │
   │  7. eval ──── compute RAGAS metrics → SQLite     │
   │        │                                         │
   │        ▼                                         │
   │       END → JSON response                        │
   └─────────────────────────────────────────────────┘
```

### 3.2 The Four Knowledge Base Corpora

The knowledge base is a **multi-corpus RAG store** — 4 separate FAISS indexes, each serving a different purpose:

| Corpus | What's Inside | Who Uses It | Purpose |
|--------|--------------|-------------|---------|
| `job_market` | Role descriptions, responsibilities by domain (SWE, DS, ML, DevOps) | JD Analyst | Understand what's standard for a given role |
| `ats_keywords` | Keyword lists per role and tech stack | JD Analyst | Identify which keywords an ATS would scan for |
| `hiring_criteria` | Rubrics, ATS scoring logic, engineering levels, interview signals | Gap Analyst | Score the candidate against real hiring standards |
| `resume_examples` | High-scoring resume bullets by domain and seniority | Rewriter | Provide examples of strong resume language |

**How they're built:**
1. Source data lives in `knowledge_base/*.json` — hand-curated JSON documents
2. `init_kb.py` reads each JSON, embeds all `content` fields, and builds FAISS indexes
3. Indexes are saved to `embeddings_store/knowledge_base/*.faiss` + `*_meta.npy`
4. In Docker, this happens at **build time** (`RUN python init_kb.py`) so the indexes are baked into the image

### 3.3 Dual Retrieval Strategy

Every request performs **two parallel retrievals** using the JD as the query:

**Source 1 — Resume Index (per-request, ephemeral):**
- The uploaded resume is chunked (300 words, 50 overlap)
- Chunks are embedded and stored in a FAISS `IndexFlatL2`
- Top-3 most JD-relevant resume chunks are retrieved
- Index is cached by MD5 hash of resume text (same resume = cache hit)

**Source 2 — Knowledge Base (persistent, shared):**
- The JD embedding queries all 4 KB corpora
- Top-2 chunks per corpus = 8 KB chunks total
- These provide grounding context for the agents

**Total context per request:** 3 resume chunks + 8 KB chunks = **11 chunks** injected across the 4 agent prompts.

---

## 4. Agent-by-Agent Breakdown

### Agent 1: Parser Agent

**File:** `backend/agents/parser_agent.py`

**Input:** Raw resume text + top-3 resume chunks (from retrieval)

**What it does:** Extracts structured data from unstructured resume text:
- Name, email
- Skills list
- Experience (title, company, duration, bullet points)
- Education (degree, institution, year, GPA)
- Certifications
- Summary

**Why use an LLM for this?** Resumes have wildly inconsistent formats. Regex-based parsers break on non-standard layouts. The LLM handles arbitrary formatting gracefully.

**Output stored in state as:** `parsed_resume` dict

---

### Agent 2: JD Analyst Agent

**File:** `backend/agents/jd_analyst_agent.py`

**Input:** Job description text + KB chunks from `job_market` and `ats_keywords` corpora

**What it does:**
- Identifies job title, company, seniority level, domain
- Separates **must-have** requirements from **nice-to-have**
- Extracts ATS keywords (exact tool/tech names)
- Lists key responsibilities

**Why the must-have/nice-to-have split matters:** A resume missing a "nice-to-have" skill is very different from missing a "must-have." The gap analyst uses this ranking to weight its scoring.

**Output stored in state as:** `parsed_jd` dict

---

### Agent 3: Gap Analyst Agent

**File:** `backend/agents/gap_analyst_agent.py`

**Input:** Parsed resume skills + parsed JD requirements + KB `hiring_criteria` chunks

**What it does (two-phase scoring):**

**Phase 1 — Semantic similarity (algorithmic):**
```python
vecs = embed_texts([" ".join(resume_skills), " ".join(must_have + keywords)])
semantic_score = cosine(vecs[0], vecs[1])
```
This produces a 0–1 score measuring how semantically close the candidate's skill set is to the JD requirements.

**Phase 2 — LLM reasoning:**
The semantic score is injected into the prompt alongside the skill lists and hiring criteria. The LLM then produces:
- A **match_score** (0–100) informed by both the algorithmic score and contextual reasoning
- **matched_skills** — skills the candidate clearly has
- **missing_skills** — each with importance level and estimated learning time
- **gap_summary** — honest 2–3 sentence assessment
- **quick_wins** — things to add without learning anything new

**Why combine algorithmic + LLM?** Pure cosine similarity can't understand nuance (e.g., "PyTorch" and "TensorFlow" are semantically close but different frameworks). Pure LLM scoring can hallucinate or be inconsistent. Combining both gives a robust, calibrated score.

**Output stored in state as:** `match_score`, `matched_skills`, `missing_skills`, `gap_summary`, `quick_wins`

---

### Agent 4: Rewriter Agent

**File:** `backend/agents/rewriter_agent.py`

**Input:** Parsed experience + JD requirements + matched/missing skills + KB `resume_examples` and `hiring_criteria` chunks

**What it does:**
- Rewrites existing resume bullets to match JD language
- Creates a tailored professional summary
- Suggests improvement tips (courses, certs, projects)

**Critical constraint:** `"Never fabricate experience."` — the rewriter only rephrases what the candidate actually did. It mirrors JD vocabulary, adds quantification where numbers already exist, and uses stronger action verbs.

**Conditional execution:** Only runs if `match_score >= 30`. Below 30, the resume is such a poor match that rewriting won't help — the response focuses on the gap analysis instead.

**Output stored in state as:** `rewritten_bullets`, `tailored_summary`, `improvement_tips`

---

## 5. Services Layer (The Engine Room)

### `embeddings.py` — Embedding Service

- **Lazy singleton:** The SentenceTransformer model is loaded on first call, cached globally
- **GPU-aware:** Automatically uses CUDA if available, falls back to CPU
- **Batch processing:** Encodes up to 64 texts at once for efficiency
- **Normalization:** All vectors are L2-normalized, enabling cosine similarity via inner product

### `corpus_builder.py` — Knowledge Base Manager

- **Build or load:** On startup, checks if FAISS indexes exist on disk. If yes, loads them. If no, builds from source JSONs.
- **Thread-safe:** Uses a threading lock for concurrent access to the cache
- **Force rebuild:** `load_all_corpora(force_rebuild=True)` re-embeds everything from source

### `retriever.py` — Dual Retrieval Engine

- **Resume cache:** Per-request FAISS indexes are cached by MD5 hash of resume text. Same resume uploaded twice = cache hit.
- **Thread-safe:** Uses a threading lock for index creation/access
- **Dual search:** Single function call returns both resume chunks and KB chunks

### `chunker.py` — Text Chunking

- Simple sliding window chunker (300 words, 50 overlap)
- Word-level splitting (not character-level) for cleaner boundaries

### `parser.py` — PDF Extraction

- Uses **PyMuPDF (fitz)** for text extraction
- **Async wrapper:** CPU-bound extraction runs in a `ThreadPoolExecutor` to avoid blocking the async FastAPI event loop
- Extracts text page-by-page and joins with newlines

### `llm.py` — LLM Dispatcher

- **Groq primary:** Routes to Groq API with `llama-3.3-70b-versatile`
- **JSON mode:** `response_format={"type": "json_object"}` ensures valid JSON output
- **Fallback:** If no API key, returns a stub response telling the user to configure it
- **SageMaker stub:** Placeholder for routing to a self-hosted Mistral-7B on SageMaker

### `eval.py` — RAGAS Evaluation Engine

- Computes 3 metrics per inference (context relevance, faithfulness, answer relevance)
- Persists to SQLite (`eval_log.db`) for historical analysis
- Provides aggregate stats and per-inference history via API endpoints

---

## 6. API & Frontend

### FastAPI Backend (`backend/app.py`)

**Lifespan events:**
- **Startup:** Loads all 4 KB corpora into memory + pre-compiles the LangGraph
- **Shutdown:** Graceful cleanup

**Middleware:**
- CORS: Allows all origins (for development)
- Metrics: Records request count, error count, and average latency

**Key endpoints:**

| Endpoint | Purpose |
|----------|---------|
| `POST /analyze` | Full agent pipeline — accepts PDF + JD, returns analysis |
| `GET /health` | Simple liveness check |
| `GET /readiness` | Verifies embedding model is loaded |
| `GET /metrics` | Request/latency counters |
| `GET /eval/metrics` | Aggregate RAGAS metrics |
| `GET /eval/history` | Per-inference eval records |
| `GET /eval/corpus/stats` | KB corpus document counts |

### Flask Frontend (`flask_frontend/app.py`)

A thin proxy layer that:
- Serves the HTML UI
- Forwards `/analyze` requests to the FastAPI backend
- Handles connection errors gracefully (Render cold start messaging)

---

## 7. DevOps & Deployment

### Docker Strategy

**Multi-stage build with baked-in indexes:**

```dockerfile
# 1. Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN playwright install chromium --with-deps

# 2. Pre-download the embedding model (avoids cold-start delay)
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('...')"

# 3. Bake KB indexes into the image (no runtime build needed)
RUN python init_kb.py

# 4. Security: non-root user
RUN useradd -m -u 1000 appuser
USER appuser
```

**Why bake indexes?** In production (ECS Fargate), containers are ephemeral. If indexes were built at startup, every container restart would re-embed all KB documents (~30 seconds). Baking them into the image eliminates this.

### Docker Compose (Local Dev)

6 services running together:

| Service | Port | Purpose |
|---------|------|---------|
| Backend (FastAPI) | 8000 | Core API |
| Frontend (Flask) | 5000 | Web UI |
| MLflow | 5001 | Experiment tracking |
| Prometheus | 9090 | Metrics collection |
| Grafana | 3000 | Dashboards |

### AWS ECS Fargate Deployment

- **4 independent Fargate services** — each scales independently
- **ECR** — Docker images pushed to AWS container registry
- **Secrets Manager** — `GROQ_API_KEY` injected at runtime (not baked into image)
- **Cloudflare Tunnel** — Stable HTTPS URL without paying for an ALB (Application Load Balancer)

### CI/CD (GitHub Actions)

On every push to `main`:
1. Run `pytest` with coverage
2. Build Docker images
3. Push to ECR
4. Rolling deploy to ECS (zero-downtime)

### SageMaker (Stub)

`sagemaker/inference.py` contains a stub for deploying Mistral-7B as a SageMaker endpoint. This would allow self-hosted LLM inference instead of relying on Groq. Currently not active — controlled by `USE_SAGEMAKER` env var.

---

## 8. Key Design Decisions & Why

### Why LangGraph instead of a simple function chain?

A simple `pipeline.py` calling functions sequentially would work, but LangGraph provides:
- **Conditional routing** (skip rewriter for low scores)
- **Shared typed state** (all agents read/write the same `ResumeState`)
- **Append-only trace** (`messages` field with `Annotated[list, add]`)
- **Future extensibility** — easy to add parallel branches, human-in-the-loop, or retry logic

### Why 4 separate KB corpora instead of one big index?

Each agent needs different context. A single index would return a mix of hiring rubrics, resume examples, and ATS keywords — most of which are irrelevant to any given agent. Separate corpora let each agent query only its relevant knowledge.

### Why Groq over OpenAI/Anthropic?

- **Speed:** Groq's LPU hardware runs inference 5-10x faster than GPU-based providers
- **Cost:** Free tier is generous enough for development and demos
- **Model quality:** Llama 3.3 70B is competitive with GPT-4 for structured extraction tasks

### Why `IndexFlatIP` for KB and `IndexFlatL2` for resumes?

Both work on normalized vectors. IP was chosen for KB because cosine similarity is the standard metric for semantic search. L2 was inherited from v3 and produces equivalent results on normalized vectors. In practice, they're interchangeable here.

### Why JSON mode for LLM outputs?

Agents need structured data (parsed skills, scored gaps, rewritten bullets). Groq's `response_format={"type": "json_object"}` guarantees valid JSON. The code also has regex fallback parsing (`re.search(r"\{.*\}", raw, re.DOTALL)`) for robustness.

### Why SQLite for eval persistence?

- Zero configuration — no database server needed
- File-based — persists across restarts, easy to back up
- Sufficient for single-worker deployments (Render free tier, local dev)
- For multi-worker production, would need PostgreSQL or similar

### Why bake KB indexes into Docker image?

ECS Fargate containers are ephemeral — they can be killed and restarted at any time. Building indexes at startup would add ~30 seconds of cold-start latency. Baking them in means the container is ready to serve immediately.

### Why async PDF parsing?

PDF text extraction with PyMuPDF is CPU-bound. Running it directly in FastAPI's async event loop would block all other requests. Using `ThreadPoolExecutor` offloads the work to a separate thread, keeping the event loop responsive.

---

> **Author:** Fadhil Muhammed N C
> **Version:** 7.0.0 · July 2026
