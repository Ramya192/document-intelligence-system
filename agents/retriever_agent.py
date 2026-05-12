import chromadb
from langchain_openai import OpenAIEmbeddings
from pathlib import Path
from settings import Settings


class RetrieverAgent:
    def __init__(self):
        self.client = chromadb.PersistentClient(path=Settings.CHROMA_PATH)
        self.collection = self.client.get_or_create_collection(
            name=Settings.COLLECTION_NAME
        )
        self.embedder = OpenAIEmbeddings(model=Settings.EMBEDDING_MODEL)

        if Settings.LLM_PROVIDER == "ollama":
            from langchain_ollama import ChatOllama
            from sentence_transformers import CrossEncoder

            self.llm = ChatOllama(
                model=Settings.OLLAMA_MODEL, base_url=Settings.OLLAMA_BASE_URL
            )
            self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        else:
            from langchain_openai import ChatOpenAI

            self.llm = ChatOpenAI(model=Settings.OPENAI_MODEL)
            self.reranker = None

    def _hyde(self, query: str) -> list[float]:
        prompt = f"""You are a bank statement analyst.
A user asked: "{query}"
Write a hypothetical bank statement excerpt that would answer this.
Be specific with amounts, dates, transaction types. Under 100 words."""
        response = self.llm.invoke(prompt)
        fake_answer = response.content
        return self.embedder.embed_query(fake_answer)

    def _generate_variants(self, query: str) -> list[str]:
        prompt = f"""Generate 3 different search query variants for:
"{query}"
Context: searching a bank statement PDF.
Return only 3 queries, one per line, no numbering, no bullets."""
        response = self.llm.invoke(prompt)
        raw = response.content
        variants = [p.strip() for p in raw.split("\n") if p.strip()]
        return variants[:3]

    def _rrf_merge(self, ranked_lists: list[list[str]], k: int = 60) -> list[str]:
        scores = {}
        for ranked_list in ranked_lists:
            for rank, doc_id in enumerate(ranked_list):
                if doc_id not in scores:
                    scores[doc_id] = 0
                scores[doc_id] += 1 / (k + rank)
        return sorted(scores, key=lambda x: scores[x], reverse=True)

    def _rerank(self, query: str, chunks: list[dict]) -> list[dict]:
        if Settings.LLM_PROVIDER != "ollama":
            import cohere

            co = cohere.Client(Settings.COHERE_API_KEY)
            results = co.rerank(
                query=query,
                documents=[c["text"] for c in chunks],
                model="rerank-english-v3.0",
            )
            for r in results.results:
                chunks[r.index]["rerank_score"] = r.relevance_score
            return sorted(chunks, key=lambda x: x["rerank_score"], reverse=True)

        pairs = [(query, chunk["text"]) for chunk in chunks]
        scores = self.reranker.predict(pairs)
        for i, chunk in enumerate(chunks):
            chunk["rerank_score"] = scores[i]
        return sorted(chunks, key=lambda x: x["rerank_score"], reverse=True)

    def retrieve(self, query: str, k: int = Settings.TOP_K) -> list[dict]:
        hyde_vector = self._hyde(query)
        variants = self._generate_variants(query)

        all_ranked_lists = []

        # HyDE search
        hyde_results = self.collection.query(
            query_embeddings=[hyde_vector], n_results=20
        )
        all_ranked_lists.append(hyde_results["ids"][0])

        # Variant searches
        for variant in variants:
            variant_vector = self.embedder.embed_query(variant)
            results = self.collection.query(
                query_embeddings=[variant_vector], n_results=20
            )
            all_ranked_lists.append(results["ids"][0])

        # RRF merge
        merged_ids = self._rrf_merge(all_ranked_lists)

        # Fetch top-20 chunks
        top20 = self.collection.get(
            ids=merged_ids[:20], include=["documents", "metadatas"]
        )
        chunks = [
            {
                "text": top20["documents"][i],
                "source": top20["metadatas"][i]["source"],
                "chunk_index": top20["metadatas"][i]["chunk_index"],
                "score": 0,
            }
            for i in range(len(top20["documents"]))
        ]

        # Rerank and return top-k
        reranked = self._rerank(query, chunks)
        return reranked[:k]
