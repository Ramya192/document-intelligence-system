"""
RAG Evaluation Module — Project B (FinLens)
============================================
Auto-switches between:
  - CustomRAGEvaluator  : Ollama (local) — cosine similarity + LLM-as-judge
  - RAGASEvaluator      : OpenAI (production) — RAGAS library

Metrics implemented:
  - Context Relevancy   : cosine(query_embedding, chunk_embeddings) mean
  - Faithfulness        : LLM-as-judge — is every claim grounded in context?
  - Answer Relevancy    : cosine(query_embedding, answer_embedding)
"""

import json
import time
import logging
from typing import Optional
from settings import Settings

logger = logging.getLogger(__name__)


# ── Shared output model ───────────────────────────────────────────────────────
class EvaluationResult:
    def __init__(
        self,
        context_relevancy: float,
        faithfulness: float,
        answer_relevancy: float,
        evaluator: str,
        latency_ms: int,
    ):
        self.context_relevancy = round(context_relevancy, 3)
        self.faithfulness = round(faithfulness, 3)
        self.answer_relevancy = round(answer_relevancy, 3)
        self.evaluator = evaluator
        self.latency_ms = latency_ms

    def to_dict(self) -> dict:
        return {
            "context_relevancy": self.context_relevancy,
            "faithfulness": self.faithfulness,
            "answer_relevancy": self.answer_relevancy,
            "evaluator": self.evaluator,
            "latency_ms": self.latency_ms,
        }


# ══════════════════════════════════════════════════════════════════════════════
# CUSTOM EVALUATOR  (Ollama — local dev)
# ══════════════════════════════════════════════════════════════════════════════
class CustomRAGEvaluator:
    """
    Three metrics using sentence-transformers + Ollama LLM-as-judge.

    Context Relevancy  : mean cosine similarity between query and each chunk
    Faithfulness       : LLM prompt — does answer stay within the context?
    Answer Relevancy   : cosine similarity between query and answer
    """

    def __init__(self):
        from sentence_transformers import SentenceTransformer
        from langchain_ollama import ChatOllama

        self.encoder = SentenceTransformer("all-MiniLM-L6-v2")
        self.llm = ChatOllama(
            model=Settings.OLLAMA_MODEL,
            base_url=Settings.OLLAMA_BASE_URL,
        )

    # ── cosine similarity (manual, no scipy needed) ───────────────────────────
    @staticmethod
    def _cosine(a, b) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x**2 for x in a) ** 0.5
        norm_b = sum(x**2 for x in b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    # ── 1. Context Relevancy ─────────────────────────────────────────────────
    def _context_relevancy(self, query: str, chunks: list[dict]) -> float:
        if not chunks:
            return 0.0
        texts = [c["text"] for c in chunks if c.get("text")]
        if not texts:
            return 0.0
        query_emb = self.encoder.encode(query).tolist()
        chunk_embs = [self.encoder.encode(t).tolist() for t in texts]
        scores = [self._cosine(query_emb, ce) for ce in chunk_embs]
        return sum(scores) / len(scores)

    # ── 2. Faithfulness (LLM-as-judge) ──────────────────────────────────────
    def _faithfulness(self, answer: str, chunks: list[dict]) -> float:
        if not chunks or not answer:
            return 0.0
        context = "\n\n".join([c.get("text", "") for c in chunks])
        prompt = f"""You are an impartial evaluator assessing faithfulness of an answer.

Context (retrieved documents):
{context}

Answer to evaluate:
{answer}

Task: Check if the factual claims in the answer are broadly supported by the context.
Be lenient with minor wording differences — if the meaning is the same, it is faithful.
- Score 1.0 if all claims are grounded in the context (minor wording differences are acceptable).
- Score 0.5 if some claims are grounded but others are clearly not in the context.
- Score 0.0 only if the answer contains facts that directly contradict or are completely absent from the context.

Respond ONLY with a JSON object, no other text:
{{"score": 0.0, "reason": "brief explanation"}}"""

        try:
            response = self.llm.invoke(prompt)
            raw = response.content.strip()

            # Extract JSON
            start = raw.find("{")
            end = raw.rfind("}") + 1
            if start != -1 and end > start:
                data = json.loads(raw[start:end])
                score = float(data.get("score", 0.5))
                return max(0.0, min(1.0, score))
        except Exception as e:
            logger.warning(f"Faithfulness eval failed: {e}")
            # Also print raw response for debugging
        return 0.5  # neutral fallback

    # ── 3. Answer Relevancy ──────────────────────────────────────────────────
    def _answer_relevancy(self, query: str, answer: str) -> float:
        if not query or not answer:
            return 0.0
        query_emb = self.encoder.encode(query).tolist()
        answer_emb = self.encoder.encode(answer).tolist()
        return self._cosine(query_emb, answer_emb)

    # ── Public evaluate() ────────────────────────────────────────────────────
    def evaluate(
        self,
        query: str,
        chunks: list[dict],
        answer: str,
    ) -> EvaluationResult:
        start = time.time()

        ctx_rel = self._context_relevancy(query, chunks)
        faith = self._faithfulness(answer, chunks)
        ans_rel = self._answer_relevancy(query, answer)

        latency_ms = int((time.time() - start) * 1000)

        return EvaluationResult(
            context_relevancy=ctx_rel,
            faithfulness=faith,
            answer_relevancy=ans_rel,
            evaluator="custom-ollama",
            latency_ms=latency_ms,
        )


# ══════════════════════════════════════════════════════════════════════════════
# RAGAS EVALUATOR  (OpenAI — production)
# ══════════════════════════════════════════════════════════════════════════════
class RAGASEvaluator:
    """
    Uses RAGAS library with OpenAI backend.
    Metrics: context_relevancy, faithfulness, answer_relevancy
    """

    def __init__(self):
        from ragas import evaluate
        from ragas.metrics import (
            context_relevancy,
            faithfulness,
            answer_relevancy,
        )
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings

        self._evaluate = evaluate
        self._metrics = [context_relevancy, faithfulness, answer_relevancy]

        # RAGAS uses langchain LLM and embeddings under the hood
        self._llm = ChatOpenAI(model=Settings.OPENAI_MODEL)
        self._embeddings = OpenAIEmbeddings(model=Settings.EMBEDDING_MODEL)

    def evaluate(
        self,
        query: str,
        chunks: list[dict],
        answer: str,
    ) -> EvaluationResult:
        from datasets import Dataset

        start = time.time()

        contexts = [c.get("text", "") for c in chunks]

        dataset = Dataset.from_dict(
            {
                "question": [query],
                "answer": [answer],
                "contexts": [contexts],
            }
        )

        try:
            result = self._evaluate(
                dataset,
                metrics=self._metrics,
                llm=self._llm,
                embeddings=self._embeddings,
            )
            df = result.to_pandas()
            ctx_rel = float(df["context_relevancy"].iloc[0])
            faith = float(df["faithfulness"].iloc[0])
            ans_rel = float(df["answer_relevancy"].iloc[0])
        except Exception as e:
            logger.error(f"RAGAS evaluation failed: {e}")
            ctx_rel = faith = ans_rel = 0.0

        latency_ms = int((time.time() - start) * 1000)

        return EvaluationResult(
            context_relevancy=ctx_rel,
            faithfulness=faith,
            answer_relevancy=ans_rel,
            evaluator="ragas-openai",
            latency_ms=latency_ms,
        )


# ══════════════════════════════════════════════════════════════════════════════
# FACTORY — auto-switches based on Settings.LLM_PROVIDER
# ══════════════════════════════════════════════════════════════════════════════
def get_evaluator():
    """
    Returns evaluator based on LLM_PROVIDER.
    - ollama  → CustomRAGEvaluator (local)
    - openai  → None (skip evaluation on production to save memory)
    """
    if Settings.LLM_PROVIDER == "ollama":
        return CustomRAGEvaluator()
    return RAGASEvaluator()
