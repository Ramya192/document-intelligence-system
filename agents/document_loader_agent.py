from fileinput import filename

from chromadb import PersistentClient
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from pathlib import Path
import pdfplumber
from settings import Settings


class DocumentLoaderAgent:
    def __init__(self):
        # HINT 1: initialise ChromaDB client
        self.client = PersistentClient(path=Settings.CHROMA_PATH)

        self.collection = self.client.get_or_create_collection(
            name=Settings.COLLECTION_NAME
        )

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=Settings.CHUNK_SIZE, chunk_overlap=Settings.CHUNK_OVERLAP
        )

        self.embedder = OpenAIEmbeddings(model=Settings.EMBEDDING_MODEL)

    def load(self, pdf_path: str, filename: str = None) -> dict:
        filename = filename or Path(pdf_path).stem  # ← this is the source of truth

        full_text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                full_text += page.extract_text() or ""

        chunks = self.splitter.split_text(full_text)
        vectors = self.embedder.embed_documents(chunks)

        # Step 4: remove duplicates
        existing = self.collection.get(
            where={"source": filename}
        )  # uses correct filename
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

        return {"status": "success", "document": filename, "chunks_stored": len(chunks)}
