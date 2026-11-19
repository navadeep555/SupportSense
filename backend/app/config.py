"""Application configuration, loaded from environment variables.

Every setting has a safe default so the app starts with zero setup.
Values in a `.env` file (or real environment variables) override the defaults.
"""
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

# backend/ folder, found from this file's location (backend/app/config.py),
# so paths work no matter which folder you run commands from.
BACKEND_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BACKEND_DIR / "data"


class Settings(BaseSettings):
    # Where to read extra values from. extra="ignore" means unknown keys in .env
    # (e.g. settings for later phases) don't crash the app.
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "SupportSense"
    app_version: str = "0.1.0"
    app_env: str = "development"  # env var: APP_ENV

    # Kept as a plain comma-separated string so it is easy to set on any host.
    cors_origins: str = "http://localhost:5173"  # env var: CORS_ORIGINS

    # ---- RAG (Phase 3) ----
    knowledge_base_dir: Path = DATA_DIR / "knowledge_base"
    index_dir: Path = DATA_DIR / "index"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    chunk_size_words: int = 150      # model reads ~256 tokens max (~180 words)
    chunk_overlap_words: int = 30
    retrieval_top_k: int = 5         # chunks returned to the caller / LLM
    retrieval_candidate_pool: int = 20  # candidates taken from each search before fusion
    # Minimum cosine similarity of the best chunk to count as "enough evidence".
    # Chosen from the retrieval evaluation (eval/run_retrieval_eval.py).
    min_evidence_similarity: float = 0.35

    # ---- LLM generation (Phase 4) ----
    llm_provider: Literal["mock", "ollama"] = "mock"   # env var: LLM_PROVIDER
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:7b"
    ollama_context_tokens: int = 8192   # our prompts are ~1-2k tokens; leaves headroom
    llm_timeout_seconds: float = 120.0  # first call also loads the model into memory
    llm_max_retries: int = 1            # extra attempts when the reply is not valid JSON

    # ---- Database (Phase 5) ----
    # SQLite = one file, no server to install. Phase 9 switches to PostgreSQL by
    # changing only this URL (env var DATABASE_URL), e.g.
    #   postgresql+psycopg://user:password@host:5432/supportsense
    database_url: str = f"sqlite:///{(DATA_DIR / 'supportsense.db').as_posix()}"

    @property
    def cors_origin_list(self) -> list[str]:
        """'a, b' -> ['a', 'b'] (blank entries removed)."""
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    """Build Settings once and reuse it (reading .env on every request is wasteful)."""
    return Settings()
