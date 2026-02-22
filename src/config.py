"""Application configuration loaded from environment variables."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    Central configuration for the application.

    Loaded from environment variables and/or .env file.
    """

    # Qdrant
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_api_key: str | None = None

    # Google Gemini
    google_api_key: str = ""

    # Embedding
    dense_embedding_model: str = "Qwen/Qwen3-Embedding-8B"
    dense_embedding_dim: int = 1024
    embed_batch_size: int = 32

    # Upload
    max_upload_size_mb: int = 50
    upload_dir: str = "uploads"

    # Qdrant batching
    upsert_batch_size: int = 64

    # Retrieval
    retrieval_k_per_collection: int = 20

    # Session
    session_ttl_hours: int = 24
    history_max_turns: int = 5

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
