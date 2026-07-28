"""Centralized application settings, loaded from environment / .env file.

Every other module should import `settings` from here rather than
reading os.environ directly, so config stays in one auditable place.
"""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- App ---
    app_env: str = Field(default="dev")
    log_level: str = Field(default="INFO")
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)

    # --- Postgres ---
    postgres_host: str = Field(default="localhost")
    postgres_port: int = Field(default=5432)
    postgres_db: str = Field(default="ragchat")
    postgres_user: str = Field(default="ragchat")
    postgres_password: str = Field(default="ragchat_dev_password")

    # --- OpenSearch ---
    opensearch_host: str = Field(default="localhost")
    opensearch_port: int = Field(default=9200)
    opensearch_use_ssl: bool = Field(default=False)
    opensearch_index: str = Field(default="chunks")

    # --- Embeddings ---
    embedding_model: str = Field(default="BAAI/bge-small-en-v1.5")
    embedding_dim: int = Field(default=384)

    # --- Qdrant Cloud ---
    qdrant_url: str = Field(default="", validation_alias="QDRANT_URL")
    qdrant_api_key: str = Field(default="", validation_alias="QDRANT_API_KEY")

    # --- LLM Provider Settings ---
    llm_provider: str = Field(default="groq")  # "groq" or "gemini"

    # --- Groq ---
    groq_api_key: str = Field(default="changeme")
    groq_model: str = Field(default="llama-3.1-8b-instant")

    # --- Google Gemini ---
    google_api_key: str = Field(default="", validation_alias="GOOGLE_API_KEY")
    gemni_api_key: str = Field(default="", validation_alias="GEMNI_API_KEY")
    gemini_model: str = Field(default="gemini-3-flash-preview")

    # --- Serper Dev (Google Search) ---
    serper_api_key: str = Field(default="")

    # --- Langfuse ---
    langfuse_public_key: str = Field(default="changeme")
    langfuse_secret_key: str = Field(default="changeme")
    langfuse_host: str = Field(default="http://localhost:3000")

    @property
    def postgres_dsn(self) -> str:
        """Async SQLAlchemy DSN (asyncpg driver)."""
        return (
            f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def opensearch_url(self) -> str:
        scheme = "https" if self.opensearch_use_ssl else "http"
        return f"{scheme}://{self.opensearch_host}:{self.opensearch_port}"


@lru_cache
def get_settings() -> Settings:
    """Cached settings singleton — import this, don't instantiate Settings() directly."""
    return Settings()


settings = get_settings()
