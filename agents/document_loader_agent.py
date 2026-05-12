from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from pathlib import Path
import pdfplumber
from qdrant_client.grpc import PayloadSchemaType
from settings import Settings


class DocumentLoaderAgent:
    def __init__(self):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=Settings.CHUNK_SIZE,
            chunk_overlap=Settings.CHUNK_OVERLAP,
        )
        self.embedder = OpenAIEmbeddings(model=Settings.EMBEDDING_MODEL)
        self._init_vector_db()

    def _init_vector_db(self):
        if Settings.VECTOR_DB == "qdrant":
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams

            self.client = QdrantClient(
                url=Settings.QDRANT_URL,
                api_key=Settings.QDRANT_API_KEY,
            )
            # Create collection if it doesn't exist
            existing = [c.name for c in self.client.get_collections().collections]
            if Settings.COLLECTION_NAME not in existing:
                self.client.create_collection(
                    collection_name=Settings.COLLECTION_NAME,
                    vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
                )
                # Create payload index on 'source' for filtering
                self.client.create_payload_index(
                    collection_name=Settings.COLLECTION_NAME,
                    field_name="source",
                    field_schema="keyword",
                )
        else:
            from chromadb import PersistentClient

            self.chroma_client = PersistentClient(path=Settings.CHROMA_PATH)
            self.collection = self.chroma_client.get_or_create_collection(
                name=Settings.COLLECTION_NAME
            )

    def load(self, pdf_path: str, filename: str = None) -> dict:
        filename = filename or Path(pdf_path).stem

        # Step 1: Extract text
        full_text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                full_text += page.extract_text() or ""

        # Step 2: Split into chunks
        chunks = self.splitter.split_text(full_text)

        # Step 3: Embed
        vectors = self.embedder.embed_documents(chunks)

        # Step 4 & 5: Store
        if Settings.VECTOR_DB == "qdrant":
            self._store_qdrant(filename, chunks, vectors)
        else:
            self._store_chroma(filename, chunks, vectors)

        return {
            "status": "success",
            "document": filename,
            "chunks_stored": len(chunks),
        }

    def _store_qdrant(self, filename: str, chunks: list, vectors: list):
        from qdrant_client.models import PointStruct, Filter, FieldCondition, MatchValue

        # Delete existing chunks for this document
        self.client.delete(
            collection_name=Settings.COLLECTION_NAME,
            points_selector=Filter(
                must=[FieldCondition(key="source", match=MatchValue(value=filename))]
            ),
        )

        # Insert new chunks
        points = [
            PointStruct(
                id=abs(hash(f"{filename}_chunk_{i}")) % (2**63),
                vector=vectors[i],
                payload={"source": filename, "text": chunks[i], "chunk_index": i},
            )
            for i in range(len(chunks))
        ]
        self.client.upsert(
            collection_name=Settings.COLLECTION_NAME,
            points=points,
        )

    def _store_chroma(self, filename: str, chunks: list, vectors: list):
        # Remove duplicates
        existing = self.collection.get(where={"source": filename})
        if existing["ids"]:
            self.collection.delete(ids=existing["ids"])

        self.collection.add(
            documents=chunks,
            embeddings=vectors,
            ids=[f"{filename}_chunk_{i}" for i in range(len(chunks))],
            metadatas=[
                {"source": filename, "chunk_index": i} for i in range(len(chunks))
            ],
        )
