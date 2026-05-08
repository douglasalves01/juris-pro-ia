"""Configuração do microserviço a partir de variáveis de ambiente e `.env`."""

from __future__ import annotations

import functools
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(".env", str(_PROJECT_ROOT / ".env")),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    database_url: str = Field(
        default="postgresql+asyncpg://jurispro:jurispro@127.0.0.1:5432/jurispro",
        validation_alias="DATABASE_URL",
    )
    database_url_sync: str | None = Field(
        default=None,
        validation_alias="DATABASE_URL_SYNC",
    )

    jwt_secret: str = Field(
        default="dev-secret-change-in-production",
        validation_alias="JWT_SECRET",
    )
    redis_url: str = Field(
        default="redis://127.0.0.1:6379/0",
        validation_alias="REDIS_URL",
    )
    queue_backend: str = Field(
        default="memory",
        validation_alias="JURISPRO_QUEUE_BACKEND",
    )

    upload_dir: Path = Field(
        default_factory=lambda: _PROJECT_ROOT / "uploads",
        validation_alias="UPLOAD_DIR",
    )
    models_dir: Path = Field(
        default_factory=lambda: _PROJECT_ROOT / "hf_models",
        validation_alias="MODELS_DIR",
    )

    access_token_expire_minutes: int = Field(
        default=15,
        validation_alias="ACCESS_TOKEN_EXPIRE_MINUTES",
    )
    refresh_token_expire_days: int = Field(
        default=7,
        validation_alias="REFRESH_TOKEN_EXPIRE_DAYS",
    )
    max_upload_bytes: int = Field(
        default=50 * 1024 * 1024,
        validation_alias="MAX_UPLOAD_BYTES",
    )

    openai_api_key: str | None = Field(
        default=None,
        validation_alias="JURISPRO_OPENAI_API_KEY",
    )
    openai_base_url: str = Field(
        default="https://api.openai.com/v1",
        validation_alias="JURISPRO_OPENAI_BASE_URL",
    )
    openai_model: str = Field(
        default="gpt-4o-mini",
        validation_alias="JURISPRO_OPENAI_MODEL",
    )
    brand_name: str = Field(
        default="JurisPro IA",
        validation_alias="JURISPRO_BRAND_NAME",
    )
    obligations_webhook_url: str | None = Field(
        default=None,
        validation_alias="JURISPRO_OBLIGATIONS_WEBHOOK_URL",
    )


@functools.lru_cache
def get_settings() -> Settings:
    return Settings()


def get_database_url_sync() -> str:
    """URL síncrona (psycopg2) para Alembic, seed e busca pgvector."""
    s = get_settings()
    if s.database_url_sync:
        return s.database_url_sync
    url = s.database_url
    if "+asyncpg" in url:
        return url.replace("+asyncpg", "+psycopg2", 1)
    if url.startswith("postgresql://") and "+psycopg" not in url:
        return "postgresql+psycopg2://" + url.removeprefix("postgresql://")
    return url


settings: Settings = get_settings()

MODELS_DIR: Path = settings.models_dir
MAX_UPLOAD_BYTES: int = settings.max_upload_bytes
CORS_ORIGINS: list[str] = ["*"]
