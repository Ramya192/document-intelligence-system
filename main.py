import json
import time
import tempfile
import os
import asyncio
import logging
from pathlib import Path
from typing import Optional, AsyncGenerator

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from agents.document_loader_agent import DocumentLoaderAgent
from agents.retriever_agent import RetrieverAgent
from agents.reasoning_agent import ReasoningAgent
from agents.validator_agent import ValidatorAgent
from agents.rag_evaluator import get_evaluator

logger = logging.getLogger(__name__)

app = FastAPI(title="FinLens — Document Intelligence API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request models ────────────────────────────────────────────────────────────
class AnalyzeRequest(BaseModel):
    query: str
    source_document: Optional[str] = None


# ── Lazy agent init ───────────────────────────────────────────────────────────
loader_agent = None
retriever_agent = None
reasoning_agent = None
validator_agent = None
rag_evaluator = None


def get_agents():
    global loader_agent, retriever_agent, reasoning_agent, validator_agent, rag_evaluator
    if loader_agent is None:
        loader_agent = DocumentLoaderAgent()
        retriever_agent = RetrieverAgent()
        reasoning_agent = ReasoningAgent()
        validator_agent = ValidatorAgent()
        rag_evaluator = get_evaluator()


# ── SSE helper ────────────────────────────────────────────────────────────────
def sse_event(step: str, status: str, elapsed_ms: int = None, data: dict = None) -> str:
    """Format a single SSE event."""
    payload = {"step": step, "status": status}
    if elapsed_ms is not None:
        payload["elapsed_ms"] = elapsed_ms
    if data:
        payload["data"] = data
    return f"data: {json.dumps(payload)}\n\n"


# ══════════════════════════════════════════════════════════════════════════════
# STANDARD ENDPOINTS (kept for backward compatibility)
# ══════════════════════════════════════════════════════════════════════════════


@app.post("/ingest")
async def ingest(file: UploadFile = File(...)):
    get_agents()
    original_stem = Path(file.filename).stem
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    result = loader_agent.load(tmp_path, filename=original_stem)
    os.remove(tmp_path)
    return result


@app.post("/analyze")
async def analyze(request: AnalyzeRequest):
    get_agents()
    chunks = retriever_agent.retrieve(request.query)
    output = reasoning_agent.reason(request.query, chunks)
    validation = validator_agent.validate(output)

    if validation["status"] == "invalid":
        return {"status": "invalid", "errors": validation["errors"]}

    # RAG evaluation
    try:
        eval_result = rag_evaluator.evaluate(
            query=request.query,
            chunks=chunks,
            answer=output.answer,
        )
        evaluation = eval_result.to_dict()
    except Exception as e:
        logger.warning(f"RAG evaluation skipped: {e}")
        evaluation = None

    return {
        "status": "valid",
        "data": validation["data"].dict(),
        "evaluation": evaluation,
    }


# ══════════════════════════════════════════════════════════════════════════════
# SSE STREAMING ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════


@app.post("/ingest/stream")
async def ingest_stream(file: UploadFile = File(...)):
    """
    Streams ingest pipeline progress as SSE events.

    Steps:
      1. PDF Extraction
      2. Text Chunking
      3. Embedding
      4. Storing in ChromaDB
    """
    get_agents()

    # Read file content before entering the generator
    file_content = await file.read()
    original_stem = Path(file.filename).stem

    async def generate() -> AsyncGenerator[str, None]:
        pipeline_start = time.time()

        # ── Step 1: PDF Extraction ────────────────────────────────────────
        yield sse_event("PDF Extraction", "running")
        await asyncio.sleep(0)  # yield control to event loop

        t0 = time.time()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file_content)
            tmp_path = tmp.name

        # Extract text only (no chunking/embedding yet)
        import pdfplumber

        full_text = ""
        with pdfplumber.open(tmp_path) as pdf:
            for page in pdf.pages:
                full_text += page.extract_text() or ""

        yield sse_event(
            "PDF Extraction", "done", elapsed_ms=int((time.time() - t0) * 1000)
        )
        await asyncio.sleep(0)

        # ── Step 2: Text Chunking ─────────────────────────────────────────
        yield sse_event("Text Chunking", "running")
        await asyncio.sleep(0)

        t0 = time.time()
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from settings import Settings as S

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=S.CHUNK_SIZE,
            chunk_overlap=S.CHUNK_OVERLAP,
        )
        chunks = splitter.split_text(full_text)

        yield sse_event(
            "Text Chunking",
            "done",
            elapsed_ms=int((time.time() - t0) * 1000),
            data={"chunks": len(chunks)},
        )
        await asyncio.sleep(0)

        # ── Step 3: Embedding ─────────────────────────────────────────────
        yield sse_event("Embedding", "running")
        await asyncio.sleep(0)

        t0 = time.time()
        from langchain_openai import OpenAIEmbeddings
        from settings import Settings as S

        embedder = OpenAIEmbeddings(model=S.EMBEDDING_MODEL)
        vectors = embedder.embed_documents(chunks)

        yield sse_event("Embedding", "done", elapsed_ms=int((time.time() - t0) * 1000))
        await asyncio.sleep(0)

        # ── Step 4: Storing in ChromaDB ───────────────────────────────────
        yield sse_event("Storing in ChromaDB", "running")
        await asyncio.sleep(0)

        t0 = time.time()
        from chromadb import PersistentClient
        from settings import Settings as S

        client = PersistentClient(path=S.CHROMA_PATH)
        collection = client.get_or_create_collection(name=S.COLLECTION_NAME)

        # Remove duplicates
        existing = collection.get(where={"source": original_stem})
        if existing["ids"]:
            collection.delete(ids=existing["ids"])

        collection.add(
            documents=chunks,
            embeddings=vectors,
            ids=[f"{original_stem}_chunk_{i}" for i in range(len(chunks))],
            metadatas=[
                {"source": original_stem, "chunk_index": i} for i in range(len(chunks))
            ],
        )
        os.remove(tmp_path)

        yield sse_event(
            "Storing in ChromaDB",
            "done",
            elapsed_ms=int((time.time() - t0) * 1000),
            data={"document": original_stem, "chunks_stored": len(chunks)},
        )
        await asyncio.sleep(0)

        # ── Pipeline complete ─────────────────────────────────────────────
        total_ms = int((time.time() - pipeline_start) * 1000)
        yield sse_event(
            "complete",
            "done",
            elapsed_ms=total_ms,
            data={"document": original_stem, "chunks_stored": len(chunks)},
        )

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.post("/analyze/stream")
async def analyze_stream(request: AnalyzeRequest):
    """
    Streams analyze pipeline progress as SSE events.

    Steps:
      1. Query Received
      2. Retrieving Chunks
      3. Reasoning (LLM)
      4. Validation
      5. RAG Evaluation
    """
    get_agents()

    async def generate() -> AsyncGenerator[str, None]:
        pipeline_start = time.time()

        # ── Step 1: Query Received ────────────────────────────────────────
        yield sse_event("Query Received", "running")
        await asyncio.sleep(0)
        yield sse_event("Query Received", "done", elapsed_ms=0)
        await asyncio.sleep(0)

        # ── Step 2: Retrieving Chunks ─────────────────────────────────────
        yield sse_event("Retrieving Chunks", "running")
        await asyncio.sleep(0)

        t0 = time.time()
        chunks = retriever_agent.retrieve(request.query)

        yield sse_event(
            "Retrieving Chunks",
            "done",
            elapsed_ms=int((time.time() - t0) * 1000),
            data={"chunks_found": len(chunks)},
        )
        await asyncio.sleep(0)

        # ── Step 3: Reasoning (LLM) ───────────────────────────────────────
        yield sse_event("Reasoning (LLM)", "running")
        await asyncio.sleep(0)

        t0 = time.time()
        output = reasoning_agent.reason(request.query, chunks)

        yield sse_event(
            "Reasoning (LLM)", "done", elapsed_ms=int((time.time() - t0) * 1000)
        )
        await asyncio.sleep(0)

        # ── Step 4: Validation ────────────────────────────────────────────
        yield sse_event("Validation", "running")
        await asyncio.sleep(0)

        t0 = time.time()
        validation = validator_agent.validate(output)

        if validation["status"] == "invalid":
            yield sse_event(
                "Validation",
                "error",
                elapsed_ms=int((time.time() - t0) * 1000),
                data={"errors": validation["errors"]},
            )
            yield sse_event("complete", "error", data={"errors": validation["errors"]})
            return

        yield sse_event("Validation", "done", elapsed_ms=int((time.time() - t0) * 1000))
        await asyncio.sleep(0)

        # ── Step 5: RAG Evaluation ────────────────────────────────────────
        yield sse_event("RAG Evaluation", "running")
        await asyncio.sleep(0)

        t0 = time.time()
        evaluation = None
        try:
            eval_result = rag_evaluator.evaluate(
                query=request.query,
                chunks=chunks,
                answer=output.answer,
            )
            evaluation = eval_result.to_dict()
        except Exception as e:
            logger.warning(f"RAG evaluation skipped: {e}")

        yield sse_event(
            "RAG Evaluation",
            "done",
            elapsed_ms=int((time.time() - t0) * 1000),
            data=evaluation,
        )
        await asyncio.sleep(0)

        # ── Pipeline complete ─────────────────────────────────────────────
        total_ms = int((time.time() - pipeline_start) * 1000)
        yield sse_event(
            "complete",
            "done",
            elapsed_ms=total_ms,
            data={
                "status": "valid",
                "data": validation["data"].dict(),
                "evaluation": evaluation,
            },
        )

    return StreamingResponse(generate(), media_type="text/event-stream")


# ── Health check ──────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok", "service": "FinLens Document Intelligence API"}
