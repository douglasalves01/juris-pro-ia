from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "LexAI API"
    debug: bool = False

    cors_origins: str = "http://localhost:3000,http://127.0.0.1:3000"

    database_url: str = "postgresql://lexai:lexai@localhost:5433/lexai"

    redis_url: str = "redis://localhost:6380/0"

    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.0-flash"

    cohere_api_key: str = ""
    cohere_embed_model: str = "embed-multilingual-v3.0"
    cohere_rerank_model: str = "rerank-multilingual-v3.0"
    rag_vector_dimensions: int = Field(default=1024, ge=8, le=4096)

    supabase_jwt_secret: str = ""
    supabase_jwt_audience: str = "authenticated"

    lexai_auth_disabled: bool = False

    rate_limit_max: int = 20
    rate_limit_window_seconds: int = 60

    lexai_system_prompt: str = (
        "Você é LexAI, assistente jurídico para o ordenamento brasileiro. "
        "Responda em português, com clareza e prudência. "
        "Quando usar o contexto normativo fornecido, indique a referência (ex.: Art. X — diploma/ano). "
        "Se o contexto for insuficiente, diga isso explicitamente."
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
