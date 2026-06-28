from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    # --- App ---
    app_name: str = "ARIA"
    debug: bool = False

    # --- Embedding Model (local, BGE) ---
    embedding_model: str = "BAAI/bge-large-en-v1.5"
    embedding_device: str = "cuda"          # "cuda" for RTX 5080, "cpu" fallback
    embedding_batch_size: int = 64

    # --- LLM (Groq API) ---
    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"
    groq_temperature: float = 0.1

    # --- Qdrant ---
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection: str = "aria_documents"
    qdrant_vector_size: int = 1024          # BGE-large output dim

    # --- PostgreSQL ---
    postgres_url: str = "postgresql+asyncpg://aria:aria_secret@localhost:5432/aria_db"

    # --- Redis ---
    redis_url: str = "redis://localhost:6379/0"

    # --- Chunking ---
    chunk_size: int = 512
    chunk_overlap: int = 64

    # --- Retrieval ---
    top_k_retrieval: int = 10
    top_k_rerank: int = 4
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # --- Auditor thresholds ---
    faithfulness_threshold: float = 0.75
    relevance_threshold: float = 0.70
    hallucination_flag_threshold: float = 0.40

    # --- Monitor ---
    drift_window_size: int = 100            # docs to compute centroid over
    drift_alert_threshold: float = 0.25    # cosine distance from centroid
    anomaly_rolling_window: int = 50        # queries for rolling score average

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()