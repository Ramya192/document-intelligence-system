from dotenv import load_dotenv
import os

load_dotenv()


class Settings:
    # ChromaDB (local dev only)
    CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_db")

    # Qdrant Cloud (production)
    QDRANT_URL = os.getenv(
        "QDRANT_URL",
        "https://8beedc9c-3131-40f3-9988-d302af440ca4.eu-west-1-0.aws.cloud.qdrant.io",
    )
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")

    # Vector DB provider switch
    VECTOR_DB = os.getenv("VECTOR_DB", "chroma")  # "chroma" or "qdrant"

    COLLECTION_NAME = os.getenv("COLLECTION_NAME", "bank_statements")
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1500))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 200))
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    COHERE_API_KEY = os.getenv("COHERE_API_KEY", "")
    TOP_K = int(os.getenv("TOP_K", 5))
