from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_env: str = "dev"

    database_url: str = "postgresql+psycopg://rag:rag@localhost:5432/rag"
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    qdrant_collection: str = "chunks"
    storage_dir: str = "./storage"
    max_upload_mb: int = 50

    embedding_model: str = "BAAI/bge-m3"
    embedding_dim: int = 1024
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    use_fp16: bool = False
    warmup_models: bool = True

    jwt_secret: str = "replace-me"
    jwt_algorithm: str = "HS256"
    access_token_minutes: int = 60
    admin_email: str = "admin@example.com"
    admin_password: str = "replace-me-with-a-long-password"

    llm_provider: str = "anthropic"  # anthropic | openai
    llm_model: str = "claude-sonnet-5-5"
    anthropic_api_key: str | None = None
    openai_base_url: str | None = None
    openai_api_key: str | None = None

    # chunking (characters, roughly 4 chars per token)
    child_chunk_chars: int = 900
    child_overlap_chars: int = 120
    parent_max_chars: int = 3500
    context_char_budget: int = 24000

    worker_poll_seconds: float = 2.0
    job_stale_minutes: int = 60


@lru_cache
def get_settings() -> Settings:
    return Settings()
