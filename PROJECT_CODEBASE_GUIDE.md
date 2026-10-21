# Project Codebase Guide

> A code-oriented tour of this repository: what it contains, how a request moves through it, why the main choices appear in the implementation, and where the implementation differs from its stated intent.
>
> **Scope and evidence.** This guide describes the files present in this checkout. Statements about behavior are based on the code and configuration, rather than assuming that the existing README or older deep-dive descriptions are current. The repository does not record the reasoning behind every small implementation choice; sections labeled **Observed choice** describe what the code does, while **Likely rationale** is an interpretation, not a documented historical fact. This is not a claim that every production environment or historical revision behaves identically.

## 1. What the project is

The repository implements a resume-to-job-description analysis service. A user uploads a PDF resume and provides job-description text. The FastAPI backend extracts text, embeds the resume and job description, retrieves related resume passages and reference passages from four small local knowledge bases, then runs four LLM-backed analysis stages orchestrated with LangGraph. It returns a fit score, skill gaps, suggestions, rewritten bullets and supporting metadata. A small Flask application serves the browser interface and proxies analysis requests to FastAPI.

The dominant execution path is the **LangGraph agent path** in `backend/agents/graph.py`. `backend/services/pipeline.py` is a separate, older single-prompt workflow; the active `/analyze` route does not call it. The code also contains an optional SageMaker inference example and AWS deployment assets, but their presence does not mean that those paths are fully wired into the active request path.

## 2. Repository map

| Path | Responsibility | Runtime status |
|---|---|---|
| `backend/app.py` | Creates FastAPI app, startup/shutdown lifespan, CORS, request counters and router registration | Active API entry point |
| `backend/routes/analyze.py` | Accepts PDF and form data, extracts PDF text, invokes graph and constructs response | Active analysis endpoint |
| `backend/routes/health.py` | Liveness and embedding-model readiness endpoints | Active |
| `backend/routes/eval.py` | Evaluation history, aggregate metrics and corpus stats endpoints | Active |
| `backend/agents/` | Shared graph state, four agents and graph wiring | Active analysis core |
| `backend/tools/retrieval.py` | Graph-facing adapter for resume indexing and dual retrieval | Active |
| `backend/tools/scraper.py` | Requests/BeautifulSoup and optional Playwright JD scraping | Implemented, but not effectively reachable from current API input contract |
| `backend/services/embeddings.py` | Lazy Sentence Transformers model and vector encoding | Active |
| `backend/services/chunker.py` | Word-window chunking | Active |
| `backend/services/corpus_builder.py` | Reads four source JSONs, builds/loads FAISS corpora and searches them | Active |
| `backend/services/retriever.py` | Per-resume FAISS cache and dual retrieval | Active |
| `backend/services/parser.py` | PyMuPDF PDF-to-text extraction through a thread pool | Active |
| `backend/services/llm.py` | Groq call dispatcher and no-key fallback | Active, with important fallback limitations |
| `backend/services/eval.py` | RAGAS-inspired calculations and SQLite persistence | Called by graph; see limitations below |
| `backend/services/pipeline.py` | Previous one-prompt analysis implementation | Legacy/unused by current route |
| `backend/models/schemas.py` | Response Pydantic models | Active |
| `backend/models/prompts.py` | Prompt template for the legacy pipeline | Not used by active graph |
| `knowledge_base/*.json` | Source documents for each reference corpus | Active input to corpus build/load |
| `flask_frontend/app.py` | Flask UI routes and HTTP proxy to FastAPI | Active if frontend is deployed/run |
| `flask_frontend/templates/index.html` | Single-page UI, styles and browser-side behavior | Active frontend asset |
| `init_kb.py` | Calls corpus loader to build indexes if absent | Image-build/local utility |
| `embeddings_store/` | Generated FAISS corpus files and metadata, if present | Runtime/build artifact; source JSON remains authoritative for rebuilds |
| `sagemaker/inference.py` | Hugging Face text-generation handler example | Standalone stub; not integrated with `call_llm` |
| `Dockerfile` | Backend image, system libraries, Python dependencies, model download and KB index build | Backend deployment definition |
| `Dockerfile.frontend` | Lightweight Flask/Gunicorn image | Frontend deployment definition |
| `compose.yaml` | Local containers: backend, frontend, MLflow, Prometheus and Grafana | Five services in this file |
| `render.yaml` | Render web-service declarations | Alternative deployment configuration |
| `ecs-task-definition.json` | Older combined ECS task definition | Coexists with newer per-service definitions; may be stale |
| `aws/task-definitions/*.json` | Individual backend, frontend, MLflow and Grafana task definitions | ECS deployment inputs |
| `aws/*.sh`, `setup-aws.sh`, `passrole-policy.json` | AWS setup/deploy/demo/stop/address helpers and IAM policy | Operational scripts; inspect before running against an account |
| `requirements.txt` | Python dependencies | Used by backend Docker build and local setup |
| `README.md`, `PROJECT_DEEP_DIVE.md` | Existing project-oriented documentation | Useful context; some claims are aspirational or inconsistent with code |
| `eval_log.db` | Existing SQLite evaluation database file | Runtime data, not source logic |
| `.github/workflows/cicd.yml` | CI/CD workflow file | Present; actual triggers and steps should be read from that file before relying on the older README summary |

Generated caches (`.pytest_cache`, `.benchmarks`, `.dist`, virtual environments) and `.git` are not application source. A local `.env` exists in the checkout; its values are intentionally not reproduced here.

## 3. System shape and boundaries

```text
Browser
  │ multipart form (PDF + JD text)
  ▼
Flask UI / proxy :5000
  │ HTTP POST /analyze
  ▼
FastAPI :8000 ─── PDF text extraction (PyMuPDF, worker thread)
  │
  ▼
LangGraph StateGraph
  scrape node → retrieval → parser agent → JD analyst → gap analyst
                                               │
                                      score >= 30?
                                       /       \
                                   rewriter    skip
                                       \       /
                                          eval
                                            │
                                            ▼
                                       JSON response

Retrieval dependencies: SentenceTransformer embeddings + FAISS
LLM dependency: Groq API when configured
Persistence: generated FAISS files + SQLite evaluation log
```

The application is a single-process Python service in the provided Docker command (`uvicorn` with one worker). Knowledge-base indexes and the embedding model are process-local objects. The resume-index cache is in-memory and process-local. Evaluation rows are stored in a local SQLite file at the working directory's `eval_log.db` path. There is no application-owned user database or account model in the inspected source.

## 4. End-to-end analysis lifecycle

### 4.1 Browser request

The browser page is a single HTML template with embedded CSS and JavaScript. The Flask route `/` renders it. Its `/analyze` handler reads the `resume` file and `job_description` form field, forwards them to the backend with `requests`, and relays the backend JSON/status. Connection and timeout errors are converted into user-facing 503/504 JSON responses. `/health`, `/eval/metrics` and `/eval/corpus` are frontend proxy routes.

The frontend backend address comes from `BACKEND_URL`, defaulting to `http://localhost:8000`. `Dockerfile.frontend` runs Gunicorn with one worker and a 200-second timeout. The page's visual design and output rendering live in the large `templates/index.html`; it is not a separate frontend framework build.

### 4.2 API validation and PDF parsing

`POST /analyze` expects multipart form fields:

| Field | Contract |
|---|---|
| `resume` | Required uploaded filename ending in `.pdf` (case-insensitive); bytes are read in full |
| `job_description` | Required form string according to FastAPI's `Form(...)` declaration |
| `jd_url` | Optional form string |

The route rejects a non-PDF filename with HTTP 422 and rejects a PDF yielding no extracted text with HTTP 422. The PDF parser opens the bytes with PyMuPDF (`fitz`), extracts text page by page, joins with newlines and strips the result. It uses a shared `ThreadPoolExecutor(max_workers=2)` to keep this CPU-bound extraction off the async event loop. The upload is not persisted to a file by this code.

There is no explicit file size/page limit, PDF password handling, OCR fallback, or content-type validation beyond the filename suffix in this route. Image-only/scanned PDFs therefore commonly extract no useful text.

### 4.3 Graph state

The route creates a `ResumeState`-shaped dictionary with input text and empty placeholders. The TypedDict in `backend/agents/state.py` names the fields shared between nodes: source texts, retrieval artifacts, parsed structures, score/gaps, rewrite results, evaluation metadata, message trace, error and latency. These type annotations aid readability but do not perform runtime validation.

The `messages` field uses `Annotated[list, operator.add]`, which tells LangGraph to append node message lists to the existing trace. Node outputs generally return `{**state, ...}` so they preserve prior values and contribute changed fields. The route initializes `latency_ms` to `None`; the active graph does not set it to elapsed processing time, so evaluation's latency field receives zero by default. The route separately measures elapsed time around `agent_graph.invoke` for the response.

### 4.4 Graph nodes and their context

1. **`scrape_jd`** checks whether `jd_url` exists *and* `jd_text` is empty. If so, it calls `scrape_jd(url)` and puts the fetched text in state. On failure, it records an error string. In the current API, `job_description` is required, so ordinary valid API requests have nonempty `jd_text`; consequently this scrape branch is not taken through that route. The frontend also sends only text. Scraper availability should therefore be treated as implemented utility code, not a functioning URL-input feature in the current form/API contract.
2. **`retrieval`** chunks the full resume using 300 words per window and 50 words of overlap; embeds each chunk; builds or reuses a per-resume FAISS index; embeds the full JD; then retrieves up to three resume chunks and up to two documents from each loaded KB corpus. Results are stored in `resume_chunks` and a corpus-name-to-list mapping in `kb_chunks`.
3. **`parser`** prompts the LLM with the first 3,000 characters of resume text and the first three retrieved resume chunks. It asks for name, email, skills, experience, education, summary and certifications. JSON is parsed directly, with a greedy brace-regex fallback after stripping Markdown fences. A missing input or parse failure sets `error` and adds a trace entry.
4. **`jd_analyst`** prompts with the first 2,000 JD characters plus up to two retrieved `job_market` and `ats_keywords` passages. It requests title, company, seniority, domain, required/preferred skills, keywords and responsibilities.
5. **`gap_analyst`** embeds the parsed resume skill strings and concatenated must-have/keyword strings, computes cosine similarity, and includes that scalar plus hiring-criteria context in an LLM prompt. If embedding fails, it silently uses `0.5`. The LLM supplies the 0–100 score and matched/missing skills, summary and quick wins. The code does not mathematically combine the semantic score with the LLM score; it only supplies the score as prompt context.
6. **Conditional routing** sends `match_score >= 30` to the rewriter, and lower or falsey scores directly to evaluation. The threshold and score bands are prompt/code constants, not calibrated from a test set in this repository.
7. **`rewriter`** uses up to the first three parsed experience entries, JD title/requirements/keywords, matched and missing skills, up to two `resume_examples` chunks and one `hiring_criteria` chunk. Its prompt instructs against fabrication and requests rewritten bullets, a tailored summary and improvement tips. This is an LLM instruction, not a deterministic verification mechanism.
8. **`eval`** flattens KB chunks, constructs a response-shaped dictionary and calls the evaluation service. Exceptions are caught and converted into an `[eval] skipped: ...` trace message; evaluation failure does not itself fail the analysis response.

An important control-flow detail: graph edges do not check `state.error`. An error set in `scrape_jd`, retrieval or an earlier agent does not automatically terminate the graph. Later nodes may run with absent inputs and set more errors or fail. The route returns HTTP 500 only if the final graph state has a truthy `error`; which error survives depends on subsequent node updates. Exceptions not caught by individual nodes can bubble out of `invoke` and are not translated by a route-level generic exception handler.

### 4.5 LLM dispatch and structured output

`backend/services/llm.py` creates a Groq client per call and requests model `llama-3.3-70b-versatile`, JSON-object response mode, and caller-provided temperature/token limit. The agents choose low temperatures (0.1 for parsing/analysis; 0.2 for rewriting) to encourage consistent structure. Each agent parses the returned text as JSON and falls back to extracting a brace-delimited object.

Dispatch rules are based on `backend/config.py`:

* `USE_SAGEMAKER=true` causes `call_llm` to raise `NotImplementedError`; it does not invoke the SageMaker module.
* Otherwise a nonempty `GROQ_API_KEY` enables Groq.
* Without a key, `call_local_llm` returns one fixed JSON object shaped like the old single-prompt response. It is not a local language model and does not match the parser/JD/gap/rewriter schemas. The graph can parse this JSON yet still have missing expected fields, so the no-key path should not be read as a useful working analysis fallback.

The active route does not use the legacy prompt template in `backend/models/prompts.py` or `call_llm` via `services/pipeline.py`.

### 4.6 Response mapping

The route uses the Pydantic `AnalyzeResponse` model. Its top-level fields are:

| Response field | Source / conversion |
|---|---|
| `score` | Graph `match_score`, defaulting to 0 |
| `missing_skills` | Extracts `skill` from dict entries; passes through non-dicts |
| `suggestions` | Graph `improvement_tips`, default empty list |
| `rewritten_bullets` | Extracts each dict's `rewritten` string; passes through non-dicts |
| `matched_skills`, `gap_summary`, `tailored_summary`, `quick_wins` | Graph fields, with empty defaults |
| `agent_trace` | Accumulated graph messages |
| `eval` | Graph evaluation metadata or null |
| `processing_time_ms` | Wall-clock time around synchronous graph invocation |

The detailed parsed resume and parsed JD are internal state and are not returned by this response model. The graph's rewrite objects (`section`, `original`, `rewritten`, `reason`) are reduced to just rewritten strings at the API boundary.

## 5. Retrieval, embeddings and knowledge data

### 5.1 Embeddings

`backend/config.py` fixes the model identifier to `sentence-transformers/all-MiniLM-L6-v2`. `get_model()` lazily creates a module-global SentenceTransformer, selecting CUDA if PyTorch reports it available and CPU otherwise. `embed_texts()` encodes a batch with batch size 64, normalized embeddings, NumPy output and float32 conversion. There is no explicit lock around first model initialization; multiple first requests in a multi-threaded process could race to initialize it. The Dockerfile downloads the model at image build time, but model availability still depends on the runtime image/cache being present.

### 5.2 Word-window chunking

`chunk_text(text, size=300, overlap=50)` splits on whitespace and advances by `size - overlap`. The retrieval wrapper uses these defaults explicitly. This is a simple, language-agnostic token-by-whitespace strategy: punctuation and semantic section boundaries are not treated specially. Empty input yields no chunks. `services/pipeline.py` has separate older settings (200/30) but is not the active graph path.

### 5.3 Knowledge-base source files and indexing

There are four JSON arrays, each expected to contain objects with a `content` field:

| Corpus name | Source | Intended prompt consumers |
|---|---|---|
| `job_market` | `knowledge_base/job_market.json` | JD analyst |
| `resume_examples` | `knowledge_base/resume_examples.json` | Rewriter |
| `ats_keywords` | `knowledge_base/ats_keywords.json` | JD analyst |
| `hiring_criteria` | `knowledge_base/hiring_criteria.json` | Gap analyst and rewriter |

The builder embeds each document as one vector (it does not chunk KB documents), builds a FAISS `IndexFlatIP`, and writes the FAISS index plus an object-array `.npy` file containing corresponding text. Vectors are normalized, so inner product ranks by cosine similarity. At load, the cache maps each corpus name to its FAISS index and text list. If both index and metadata files exist, they are loaded; if either is absent, the corpus is rebuilt from JSON. `force_rebuild=True` removes existing paths and rebuilds. `init_kb.py` calls `load_all_corpora()`; despite its message saying “indexed successfully,” it may simply load already-built files.

Paths in the builder are relative to the process working directory (`knowledge_base` and `embeddings_store/knowledge_base`). Running from the project root is therefore assumed. The Dockerfile sets `/app` as its working directory and copies the source JSON before building.

### 5.4 Per-resume index and cache

The retriever hashes the complete extracted resume text using MD5 and stores `(FAISS IndexFlatL2, chunks)` in a module-global dictionary keyed by that hash. A lock guards cache lookup/creation. The index is built from normalized vectors; squared L2 ranking on unit vectors yields the same ranking as cosine similarity. The cache has no size bound, TTL or eviction. Thus repeated identical resume text reuses an index within a process, while varied uploads accumulate entries until process exit. The key is a cache key, not authentication or integrity protection.

### 5.5 What retrieval does and does not do

Each JD embedding queries the resume index and every loaded KB corpus independently. The code returns text only; similarity scores are discarded. The four corpus results are not globally reranked against one another. Per-agent prompts selectively use certain corpora, but the retrieval call searches every corpus on every request. Context is further truncated in each prompt by the agent-specific character limits described above. This is local vector retrieval over the supplied static JSON, not a continuously updated external market index.

## 6. Evaluation and observability

### 6.1 Evaluation calculations

`backend/services/eval.py` calls its measurements “RAGAS-inspired.” It does not call the RAGAS package. It calculates:

* **Context relevance:** mean cosine similarity between the JD vector and embeddings of all retrieved resume and KB chunks.
* **Faithfulness:** Jaccard overlap of lowercase whitespace-token sets from retrieved context and a response string. This is a lexical proxy, not fact checking or entailment.
* **Answer relevance:** cosine similarity between the JD vector and an embedding of the response string.

It then writes a row into `eval_records` in `eval_log.db`. The schema records Unix timestamp, three metric values, counts of resume/KB chunks, LLM score and latency. Each read/write opens a SQLite connection and ensures the table exists. The history endpoint caps requested rows at 200; the route accepts 1–200.

**Known implementation mismatch:** `compute_and_log()` joins `missing_skills`, `suggestions`, and `rewritten_bullets` as strings. The active graph passes `missing_skills` entries that may be dictionaries and `rewritten_bullets` entries that are dictionaries. Python `str.join` raises `TypeError` on those shapes. The graph catches the exception and marks evaluation skipped. This makes evaluation persistence conditional and likely to fail for normal structured outputs. Also, `eval_metadata` is set only when persistence succeeds, and `latency_ms` remains zero because the graph never updates that state field. The calculations should not be interpreted as guaranteed metrics for each API call.

### 6.2 API counters and logs

`backend/core/metrics.py` contains process-local integer/float counters. The HTTP middleware increments request count and latency after successful middleware handling; it increments error count only when an exception escapes `call_next`. Ordinary HTTP error responses (for example a route returning 422/500) are still responses and are counted as requests, not necessarily as errors. `/metrics` returns these counters; it is not a Prometheus exposition format despite Prometheus being present in Compose. No `/metrics` scrape-format endpoint is implemented in this FastAPI file.

`backend/core/logging.py` configures standard Python logging at INFO. The graph's trace is returned to the API caller. MLflow is configured in Compose/task definitions and an environment setting exists, but the inspected active analysis path does not log runs to MLflow. Prometheus and Grafana containers are declared in Compose, but the backend-to-Prometheus scrape integration/configuration should be verified from the actual deployment files; the application counter endpoint alone is not a Prometheus-format exporter.

## 7. API and user interface reference

| Method/path | Behavior |
|---|---|
| `GET /` (FastAPI) | Service-name JSON |
| `POST /analyze` | PDF + JD analysis described above |
| `GET /health` | Always returns `{"status":"ok"}` if process can answer |
| `GET /readiness` | Calls lazy embedding model loader and returns ready/not_ready/error |
| `GET /metrics` | Process-local request/error/average-latency counters as JSON |
| `GET /eval/metrics` | Aggregate SQLite metrics, or no-record message |
| `GET /eval/history?limit=50` | Recent rows, with 1–200 query validation |
| `GET /eval/corpus/stats` | Loaded corpus counts/index classes, or not-loaded status |

The liveness endpoint does not check the database, Groq credentials, corpus cache, or graph. Readiness does initialize/load the embedding model, so it can be expensive on a cold process; it does not verify Groq access. App startup tries to load corpora and compile the graph, but catches/logs failures and continues to yield the application, rather than refusing startup.

The Flask UI is a convenience proxy, not an authentication or authorization boundary. The backend is callable directly. The API configures permissive CORS (`allow_origins=["*"]`, all methods/headers), and no authentication, rate limiting, request-size policy or user isolation is visible in the inspected code.

## 8. Configuration and dependencies

| Setting | Default/behavior | Read in |
|---|---|---|
| `GROQ_API_KEY` | Empty means fixed stub response; nonempty enables Groq unless SageMaker selected | `backend/config.py` |
| `USE_SAGEMAKER` | False; case-insensitive string check for `true` | `backend/config.py` |
| `SAGEMAKER_ENDPOINT` | Empty; declared but unused by active dispatcher | `backend/config.py` |
| `MLFLOW_TRACKING_URI` | `http://localhost:5001` | Configured in `backend/config.py`; active path does not use it |
| `BACKEND_URL` | `http://localhost:8000` | Flask frontend |
| `PORT` | 5000 for direct Flask development invocation | Flask frontend |

`load_dotenv()` runs when `backend.config` imports, loading `.env` when available. Do not commit secret values. `requirements.txt` includes FastAPI/Uvicorn/multipart/Pydantic; LangGraph and LangChain Core; Groq; Transformers/Sentence Transformers/FAISS/NumPy/Torch; PyMuPDF; Requests/BeautifulSoup/Playwright; MLflow/Prometheus client; Flask/Gunicorn; boto3 and python-dotenv. Several dependencies are unpinned, so repeat builds can resolve different versions over time. The backend image also installs browser system packages and Playwright Chromium, even though active JD scraping is gated off by the request contract.

The root `.env.example` referenced by the README is not present in the enumerated project files in this checkout; create a local `.env` manually or use environment variables, without copying secrets into documentation or version control.

## 9. Build, run and deployment configurations

### 9.1 Local Python

From the repository root, create/activate a virtual environment, install `requirements.txt`, set `GROQ_API_KEY` in the environment (or a local ignored `.env`), then run:

```powershell
python init_kb.py
uvicorn backend.app:app --reload --port 8000
```

Run the Flask app separately if desired:

```powershell
$env:BACKEND_URL = "http://localhost:8000"
python -m flask --app flask_frontend.app run --port 5000
```

The backend can build missing corpora during startup, so explicit `init_kb.py` is not always needed. It is useful to perform that work ahead of serving. First embedding-model use downloads model artifacts if unavailable and may require network access.

### 9.2 Backend Docker image

`Dockerfile` uses `python:3.12-slim`, installs native libraries (including libraries needed by numerical/Playwright dependencies), installs Python requirements, installs Playwright Chromium, copies backend and KB sources, pre-downloads `all-MiniLM-L6-v2`, and runs `init_kb.py` during image build. It exposes port 8000, creates UID 1000 `appuser`, switches to that user, defines a curl `/health` check, and starts one Uvicorn worker. Baking the KB indexes/model into the image trades larger build/image size and build-time network/model work for less startup indexing work. The health check only checks liveness.

### 9.3 Compose

`compose.yaml` declares five services:

| Service | Host port | Role |
|---|---:|---|
| `backend` | 8000 | FastAPI image; receives Groq key and MLflow URI; depends on MLflow |
| `frontend` | 5000 | Flask image; points to backend service DNS |
| `mlflow` | 5001 | MLflow 2.13 server with SQLite backend and artifact directory |
| `prometheus` | 9090 | Prometheus 2.51 with mounted config and seven-day retention |
| `grafana` | 3000 | Grafana 10.4 with mounted datasource provisioning |

Named volumes are declared for embeddings, evaluation DB, MLflow, Prometheus and Grafana. The backend uses `depends_on: mlflow`, but Compose's simple dependency does not by itself establish application-level tracking behavior. The backend MLflow URI is supplied, though active code does not log MLflow runs. In this YAML the evaluation volume is mounted at `/app/eval_log.db`; verify Docker volume semantics and persistence expectations before treating it as a reliable database backup strategy. Grafana's admin password is configured as `admin`, appropriate only for disposable local use and not a public deployment.

### 9.4 Render

`render.yaml` defines two Docker web services, backend and frontend. It asks Render to provide `GROQ_API_KEY` as a dashboard secret and sets `USE_SAGEMAKER=false`; frontend obtains backend host from Render service metadata. The deployment file does not configure a persistent disk for SQLite or generated files, so local file persistence across replacement instances should not be assumed. Render's health paths are `/health`; the frontend implements that as a backend proxy.

### 9.5 AWS assets

There are two ECS layouts in the repository:

* `aws/task-definitions/{backend,frontend,mlflow,grafana}.json` and `aws/deploy-all-services.sh` describe four separate services in cluster `dis-v7-cluster`.
* Root `ecs-task-definition.json` describes four containers inside one task family `dis-v7`, and `setup-aws.sh` registers that combined task and a service.

These are distinct deployment approaches and are not interchangeable without reconciling service names, networking and environment variables. The per-service definitions contain account, region, image and resource assumptions. Backend secrets are referenced by a Secrets Manager ARN. The frontend task definition uses `localhost:8000`, which only works if frontend and backend share a task/network namespace; it does not match the separate-service deployment where `BACKEND_URL` needs a reachable backend address. The root combined task does use localhost networking between containers, but its MLflow URI and volume assumptions need to be checked against the runtime configuration. Root task definitions include a hard-coded account/region; treat them as environment-specific templates.

`setup-aws.sh` creates log groups, writes a Groq secret, creates/updates IAM role/policy, edits the root ECS task-definition file with an account ID, registers it, creates a cluster/security group, opens ports 8000/5000/5001/3000 to `0.0.0.0/0`, and creates a service. This has substantial external side effects and changes the JSON file. Review the script and account values before using it. `aws/deploy-all-services.sh` registers and creates/updates four services. `aws/stop-all.sh` scales them to zero. `aws/get-urls.sh` prints task public IP URLs; those addresses are ephemeral. `aws/demo.sh` starts backend/frontend and tails logs, including references to Cloudflare tunnel output not clearly established by the inspected task definitions. `passrole-policy.json` is an IAM policy artifact.

The README mentions GitHub Actions, Cloudflare Tunnel and a four-service ECS deployment; the mere presence of an Actions workflow or scripts is not evidence that all those pieces are currently integrated. Verify `.github/workflows/cicd.yml`, account setup, networking and current images before treating deployment claims as live operational guarantees.

### 9.6 SageMaker example

`sagemaker/inference.py` defines `model_fn` that loads `mistralai/Mistral-7B-Instruct-v0.2` through a Transformers text-generation pipeline and `predict_fn` that accepts `data["prompt"]`. It returns generated text. It is not imported by `backend/services/llm.py`, does not invoke a SageMaker Runtime endpoint, and is not a deployed endpoint definition. Setting `USE_SAGEMAKER=true` currently raises `NotImplementedError` in active calls.

## 10. Architectural choices visible in the code

These are observations and plausible explanations, not recovered design records.

| Observed choice | What it buys / likely rationale | Cost or consequence |
|---|---|---|
| LangGraph `StateGraph` with typed shared state | Makes agent stages and one conditional branch explicit; a shared dictionary carries intermediate results | State type is not runtime validation; errors do not short-circuit automatically |
| Four semantically separated KB indexes | Retrieval can return domain-specific contexts and prompts choose relevant corpora | Every request searches all four; documents are static and similarity scores are discarded |
| `IndexFlatIP` on normalized KB vectors | Exact nearest-neighbor cosine-like ranking without approximate-index tuning | Linear scan, suitable for the small corpus size implied by bundled JSON, not necessarily large collections |
| `IndexFlatL2` on normalized resume vectors | Simple exact per-resume lookup; on unit vectors ranking matches cosine ordering | Name/metric differ but ranking equivalence depends on normalization being preserved |
| Local SentenceTransformer embeddings | Avoids an external embedding API and produces a shared vector space for query and corpus | Model download, CPU/memory cost and cold initialization; no model version metadata on saved indexes |
| Groq JSON-object mode and JSON prompts | Agent code needs structured fields and provider format constrains output | Still relies on parsing/expected keys; no schema validation/repair or retry loop |
| PDF text extraction via thread pool | Avoids doing CPU-bound PDF parsing directly on the async event loop | Thread pool has two workers; LLM/graph invocation itself is synchronous inside an async route |
| SQLite eval log | No database service required for a small single-instance demonstration | Local-file persistence and concurrent/multi-instance behavior are limited; metrics helper has shape mismatch |
| One Uvicorn/Gunicorn worker in Docker | Keeps module-global caches and SQLite deployment simple and limits memory duplication | Throughput is constrained; caches are not shared between replicas |
| Docker image includes model and built KB | Moves first-run model download and indexing work into build | Larger build and image; stale indexes can persist until image rebuilt; saved-index/model compatibility is not checked |
| Rewriter skipped below score 30 | Avoids spending another LLM call on low-fit cases, as intended by code comments | Threshold is a fixed heuristic with no demonstrated validation |
| Flask as thin proxy | Simple server-rendered frontend can call backend without browser CORS handling for same-origin UI | Flask is not authentication; backend API remains separately accessible |

## 11. Important discrepancies and edge cases

This section is intentionally direct: these details affect whether the system works as described.

1. **JD URL scraping is gated off by required text.** `job_description` is mandatory and the graph scrapes only if it is empty. Neither current frontend nor route exposes a URL-only submission path.
2. **Evaluation often skips on valid structured outputs.** Dict-valued skill gaps and rewrite objects are fed into string joins in `compute_and_log`; the graph catches this and does not produce eval metadata for that request.
3. **No-key mode is not a functional local model.** It returns a legacy JSON shape that does not satisfy each agent's expected output schema.
4. **SageMaker selection fails intentionally.** The flag routes to an unimplemented branch; the standalone inference example is not connected.
5. **Graph errors are not terminal edges.** Nodes continue along fixed edges after state errors, potentially masking the original failure or triggering later exceptions.
6. **Graph timing metadata is not filled.** The response's own processing time is measured, but `state.latency_ms` remains `None`, so eval latency is zero when eval succeeds.
7. **Evaluation faithfulness is only token overlap.** It cannot determine whether claims are supported, and it can score lexical overlap rather than truth.
8. **No output schema validation at agent boundaries.** JSON decoding is checked, but required keys, types, numeric score range and list item shape are not validated before use.
9. **Resume index cache grows without eviction.** Every unique resume text remains in the process cache until process restart.
10. **Startup errors are logged, not fatal.** The app may report healthy while KB initialization or graph compilation failed; `/health` is liveness only.
11. **The API performs synchronous graph/LLM work inside an async endpoint.** PDF extraction is offloaded, but graph invocation and synchronous Groq calls can occupy the event loop while running.
12. **Prompt excerpts are truncated differently by stage.** Parser sees 3,000 resume characters plus retrieved chunks; JD analysis sees 2,000 JD characters; full JD is embedded/retrieved before those prompt truncations. Long documents can therefore be represented differently at retrieval and reasoning time.
13. **The browser's title/content and docs may overstate API guarantees.** UI affordances and project docs should be compared with the endpoint response and routing behavior, which are authoritative for current backend behavior.
14. **Multiple deployment definitions disagree.** Separate ECS services use localhost for frontend-to-backend in at least one task file, while separate tasks need service DNS/public address; root combined ECS config and split services are separate strategies.
15. **Metrics names imply integrations that are incomplete.** An MLflow URI exists and containers are defined, but no active run logging was found; the JSON `/metrics` route is not standard Prometheus exposition.

## 12. Security, privacy and operational notes

* Resume contents are sent to the configured LLM provider in prompts. The code does not redact names, email addresses or other personal information before the call. Users/operators must account for their provider and data-handling policies.
* The code has no user authentication, access control, rate limiting or explicit upload limits in the inspected routes.
* CORS allows every origin. This is convenient for development, but broad for a public API.
* The JD scraper uses `requests.get` for caller-provided URLs and follows the normal Requests behavior. If a URL-only route is later exposed to untrusted users, outbound-request controls and network restrictions need deliberate design.
* The AWS setup script opens service ports publicly and includes account/role/resource operations. Treat it as an infrastructure-changing script, not a harmless local setup command.
* `eval_log.db` and resume-index caches can contain derived operational information; the current cache stores resume text chunks in process memory. No retention/erasure workflow is implemented.
* The rewriter's anti-fabrication rules are prompts, not a guarantee. Results need human review before being used in an application.

## 13. How to extend the project safely

* **Add a KB document:** preserve the JSON list/object/`content` shape, then rebuild indexes (`python init_kb.py` only creates missing indexes; use the builder's `force_rebuild=True` from a controlled script or remove generated index files deliberately to reflect edits). Rebuild the Docker image if indexes are baked into it.
* **Change embedding model:** update the model identifier, rebuild every corpus index, and clear/restart process-local resume caches. Mixing vector dimensions/models is invalid; the saved indexes contain no model compatibility manifest.
* **Add/change an agent output:** update its prompt, state field, parsing/validation, consumers, API response mapping and evaluation serialization together. Do not rely on syntactically valid JSON alone.
* **Expose URL-based JD input:** make text optional at API and frontend boundaries, validate URL scheme/host and outbound access, define behavior when both text and URL are supplied, and choose whether to support the Playwright path explicitly.
* **Repair evaluation:** normalize nested values into strings or store structured JSON separately, compute latency around the graph, and describe the lexical faithfulness proxy accurately or replace it with a suitable evaluator.
* **Scale beyond one process:** plan shared persistent storage/database, bounded or external resume caching, concurrency-safe embedding initialization, model memory per replica, and observability exporters before adding workers/replicas.
* **Make providers interchangeable:** implement a real SageMaker Runtime client behind the dispatcher, with endpoint configuration, timeout/error policy and response normalization matching each agent's contract.
* **Deployment changes:** choose either combined ECS task networking or separate services, then align `BACKEND_URL`, service discovery, health checks, secrets, storage and task definitions with that one topology.

## 14. Suggested reading order for a new maintainer

1. `README.md` for the product's intended behavior and basic commands.
2. `backend/routes/analyze.py` for the actual request contract and response mapping.
3. `backend/agents/graph.py` and `backend/agents/state.py` for runtime orchestration.
4. Agent files in parser → JD analyst → gap analyst → rewriter order.
5. `backend/services/embeddings.py`, `chunker.py`, `retriever.py` and `corpus_builder.py` for retrieval mechanics.
6. `backend/services/eval.py` and `backend/app.py` for persistence and operational endpoints.
7. `flask_frontend/app.py` and `templates/index.html` for user-facing behavior.
8. `Dockerfile`, `compose.yaml`, `render.yaml` and then one chosen AWS deployment path.
9. `backend/services/pipeline.py`, `backend/models/prompts.py` and `sagemaker/inference.py` as legacy/reference or incomplete paths, not as the active analysis implementation.

## 15. Glossary

* **Agent:** A specialized LLM call with a task-specific prompt and state inputs; in this code, the term does not imply an autonomous tool-using loop.
* **Corpus:** One named collection of text documents plus a FAISS index and text metadata.
* **Embedding:** A numeric vector representing text for semantic similarity search.
* **FAISS:** A library for vector nearest-neighbor indexes; this project uses exact flat indexes.
* **RAG:** Retrieval-augmented generation: retrieve relevant passages and include them as context for an LLM. Here retrieved context is selectively inserted into agent prompts.
* **StateGraph:** LangGraph's directed graph of nodes and transitions; state is passed through the nodes.
* **ATS:** Applicant tracking system; the bundled ATS keyword corpus is reference text, not integration with a real ATS.
* **Liveness/readiness:** Liveness asks whether the process responds; readiness should indicate whether it can serve. Current liveness is minimal, and readiness only forces/checks embedding model loading.
