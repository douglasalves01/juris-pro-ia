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

    rabbitmq_url: str = Field(
        default="amqp://guest:guest@rabbitmq:5672//",
        validation_alias="RABBITMQ_URL",
    )

    upload_dir: Path = Field(
        default_factory=lambda: _PROJECT_ROOT / "uploads",
        validation_alias="UPLOAD_DIR",
    )
    models_dir: Path = Field(
        default_factory=lambda: _PROJECT_ROOT / "hf_models",
        validation_alias="MODELS_DIR",
    )
    max_upload_bytes: int = Field(
        default=50 * 1024 * 1024,
        validation_alias="MAX_UPLOAD_BYTES",
    )

    gemini_api_key: str | None = Field(
        default=None,
        validation_alias="JURISPRO_GEMINI_API_KEY",
    )
    gemini_model: str = Field(
        default="gemini-2.0-flash",
        validation_alias="JURISPRO_GEMINI_MODEL",
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


settings: Settings = get_settings()

MODELS_DIR: Path = settings.models_dir
MAX_UPLOAD_BYTES: int = settings.max_upload_bytes
CORS_ORIGINS: list[str] = ["*"]
