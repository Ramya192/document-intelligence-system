---
title: FinLens API
emoji: 🏦
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---

# FinLens — Document Intelligence System

> **Project B · Agentic AI Portfolio · Ramya Priyanka A**

A production-grade BFSI document intelligence platform that ingests bank statement PDFs, answers natural language queries using a RAG pipeline, detects financial anomalies, and evaluates response quality using RAG metrics — all with a real-time streaming pipeline UI.

---

## Live Demo

| Service | URL |
|---------|-----|
| Streamlit UI | _[Deploy link — coming soon]_ |
| FastAPI Backend | _[Deploy link — coming soon]_ |
| GitHub | [github.com/Ramya192/document_intelligence_system](https://github.com/Ramya192/document_intelligence_system) |

---

## What It Does

A user uploads a bank statement PDF (HDFC, SBI, ICICI, Axis, Kotak). The system:

1. Extracts and chunks the text, generates embeddings, stores in ChromaDB
2. Accepts a natural language query ("What are the large withdrawals in March?")
3. Retrieves relevant chunks, reasons over them using an LLM, validates the output
4. Detects anomalies (suspicious withdrawals, unusual patterns)
5. Evaluates the response quality using RAG metrics (Context Relevancy, Faithfulness, Answer Relevancy)
6. Streams every pipeline step to the UI in real time with per-step timing

---

## ABCDEF Framework

### A — Agentic Architecture

Four specialised agents, each with a single responsibility:

| Agent | Responsibility |
|-------|---------------|
| `DocumentLoaderAgent` | PDF extraction → chunking → embedding → ChromaDB storage |
| `RetrieverAgent` | Semantic search over ChromaDB using OpenAI embeddings |
| `ReasoningAgent` | LLM-powered query answering with structured JSON output |
| `ValidatorAgent` | Pydantic validation of LLM output before returning to client |

Plus a dual-backend RAG evaluator:
- **`CustomRAGEvaluator`** (Ollama/local) — cosine similarity + LLM-as-judge
- **`RAGASEvaluator`** (OpenAI/production) — RAGAS library

Auto-switches via `Settings.LLM_PROVIDER` environment variable.

### B — BFSI Context

Designed specifically for BFSI use cases:

- Bank statement PDF ingestion (examples - HDFC, SBI, ICICI, Axis, Kotak formats planned in v2)
- Transaction extraction with INR amounts, dates, and transaction types
- Anomaly detection for suspicious patterns (large withdrawals, balance overdraft)
- Salary credit / advance detection
- UPI, NEFT, ATM transaction classification

### C — Core Tech Stack

| Layer | Technology |
|-------|-----------|
| LLM (local) | `llama3.1:8b` via Ollama |
| LLM (production) | `gpt-4o-mini` via OpenAI API |
| Embeddings | `text-embedding-3-large` (OpenAI) |
| Vector DB | ChromaDB (PersistentClient) |
| Framework | LangChain, FastAPI, Streamlit |
| RAG Evaluation | Custom (cosine + LLM-as-judge) / RAGAS |
| PDF Extraction | pdfplumber |
| Chunking | RecursiveCharacterTextSplitter |
| Testing | pytest + httpx (34 tests) |
| Deployment | Streamlit Cloud + Render |

### D — Design Decisions

**Why ChromaDB over Pinecone/Qdrant?**
Local PersistentClient requires zero infrastructure for development. Switching to a hosted vector DB in v2 is a one-line change in `settings.py`.

**Why custom RAG evaluator instead of RAGAS only?**
RAGAS requires OpenAI by default — not suitable for local Ollama development. The dual-backend pattern (Custom for dev, RAGAS for prod) mirrors real MLOps practice and keeps local inference cost-free.

**Why SSE streaming instead of polling?**
Server-Sent Events give true real-time step visibility with per-step timing. Each pipeline step (retrieval, LLM, validation, evaluation) streams its status and elapsed time as it completes — users see exactly where time is spent.

**Why Pydantic for LLM output validation?**
llama3.1:8b returns inconsistent JSON — sometimes `answer` is a list, `confidence` is a string, or `description` is null. The `ValidatorAgent` + repair logic in `_parse_ollama_response()` handles all known failure modes defensively.

### E — Evaluation

**RAG Metrics (per query):**

| Metric | Method | Typical Score |
|--------|--------|---------------|
| Context Relevancy | Cosine(query, chunks) via `all-MiniLM-L6-v2` | 0.27–0.38 |
| Faithfulness | LLM-as-judge: is every claim grounded in context? | 0.0–1.0 |
| Answer Relevancy | Cosine(query, answer) | 0.50–0.67 |

**Known limitations:**
- Context Relevancy is low (0.27–0.38) because `all-MiniLM-L6-v2` cosine similarity between short queries and long bank statement chunks is naturally low — this is expected behaviour, not a retrieval failure.
- Faithfulness was initially 0.00 due to the LLM penalising minor wording differences ("Rent" vs "Rent payment NEFT"). Fixed by softening the faithfulness prompt.
- RAG Evaluation adds ~27–32 seconds latency on CPU-only Ollama.

**Test coverage:** 34 pytest tests across Health, Ingest, Analyze, Stream, and Edge Cases. Session-scoped fixtures reduce total test runtime to ~8 minutes on CPU Ollama.

### F — Future Enhancements (v2 Roadmap)

- **Multi-bank format detection** — `format_detector.py` for HDFC/SBI/ICICI/Axis/Kotak PDFs using `pdfplumber.extract_table()` with per-format normalisation
- **Kaggle dataset integration** — [Bank Transaction Dataset for Fraud Detection](https://www.kaggle.com/datasets/valakhorasani/bank-transaction-dataset-for-fraud-detection) for realistic test data generation via `generate_statement.py`
- **Ground truth evaluation** — labeled Q&A pairs to unlock Context Precision/Recall, Answer Correctness, MRR
- **RAGAS in production** — switch to full RAGAS evaluation when deployed with OpenAI
- **Docker Compose** — single command to run FastAPI + ChromaDB + Ollama locally
- **LoRA fine-tuning** — domain-specific fine-tuning of the base LLM on BFSI transaction language

---

## Project Structure

```
document_intelligence_system/
├── agents/
│   ├── document_loader_agent.py   # PDF → chunks → embeddings → ChromaDB
│   ├── retriever_agent.py         # Semantic search
│   ├── reasoning_agent.py         # LLM reasoning + JSON repair
│   ├── validator_agent.py         # Pydantic validation
│   └── rag_evaluator.py           # CustomRAGEvaluator + RAGASEvaluator
├── tests/
│   └── test_api.py                # 34 pytest tests
├── main.py                        # FastAPI app + SSE streaming endpoints
├── app.py                         # Streamlit UI
├── settings.py                    # Environment config
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── .env.example
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check |
| `POST` | `/ingest` | Ingest PDF (standard) |
| `POST` | `/ingest/stream` | Ingest PDF with SSE step streaming |
| `POST` | `/analyze` | Query document (standard) |
| `POST` | `/analyze/stream` | Query document with SSE step streaming |

### SSE Event Format

```json
{"step": "Reasoning (LLM)", "status": "running"}
{"step": "Reasoning (LLM)", "status": "done", "elapsed_ms": 52560}
{"step": "complete", "status": "done", "elapsed_ms": 132110, "data": {...}}
```

---

## Quickstart

### Prerequisites

- Python 3.13+
- [Ollama](https://ollama.ai) with `llama3.1:8b` pulled
- OpenAI API key (for embeddings)

### Installation

```bash
git clone https://github.com/Ramya192/document_intelligence_system
cd document_intelligence_system
pip install -r requirements.txt
```

### Environment Setup

```bash
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

### Run Locally

```bash
# Terminal 1 — Start Ollama
ollama run llama3.1:8b

# Terminal 2 — Start FastAPI
uvicorn main:app --reload --port 8000

# Terminal 3 — Start Streamlit
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501)

### Run Tests

```bash
pytest tests/test_api.py -v
```

---

## Self-Evaluation Rubric

| Criterion | Score | Notes |
|-----------|-------|-------|
| Agent architecture (4 agents + evaluator) | 10/10 | Clear separation of concerns |
| BFSI relevance | 9/10 | Bank statements, INR, anomaly detection |
| RAG pipeline quality | 8/10 | Retrieval + reranking + evaluation |
| LLM output robustness | 8/10 | JSON repair for 5 known failure modes |
| Streaming UI | 9/10 | SSE with per-step timing, pipeline boxes |
| RAG evaluation | 8/10 | Dual backend, 3 metrics, score display |
| Test coverage | 9/10 | 34 tests, session fixtures, edge cases |
| Code quality | 8/10 | Type hints, Pydantic models, lazy loading |
| **Total** | **69/80** | |

---

## Known Limitations

- **Retrieval latency on CPU** — llama3.1:8b inference takes 50–120s on CPU-only hardware (no GPU). Production deployment uses gpt-4o-mini which is significantly faster.
- **Single chunk on small PDFs** — `CHUNK_SIZE=1500` means small bank statements fit in 1 chunk. This is intentional to avoid mid-transaction splits.
- **RAG evaluation adds latency** — ~27–32s for faithfulness LLM-as-judge call on CPU Ollama. Acceptable for demo; production uses RAGAS with OpenAI.
- **source_document from LLM** — the source document name is extracted by the LLM from document content, not from file metadata. Accurate for well-formatted PDFs.

---

## Author

**Ramya Priyanka A**  
Agentic AI Developer | BFSI + AI Transition  
[GitHub: Ramya192](https://github.com/Ramya192)
