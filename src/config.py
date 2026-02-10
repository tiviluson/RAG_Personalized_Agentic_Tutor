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
    dense_embedding_model: str = "models/gemini-embedding-001"
    dense_embedding_dim: int = 3072
    embed_batch_size: int = 100

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
