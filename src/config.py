"""Application configuration loaded from environment variables."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    Central configuration for the application.

    Loaded from environment variables and/or .env file.
    """

    # Google Gemini
    google_api_key: str = ""

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
