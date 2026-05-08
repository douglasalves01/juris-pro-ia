"""Aplicação Celery (broker e backend Redis)."""

from __future__ import annotations

from celery import Celery


def _make_celery() -> Celery:
    from api.config import get_settings

    s = get_settings()
    app = Celery(
        "jurispro",
        broker=s.redis_url,
        backend=s.redis_url,
    )
    app.conf.update(
        task_serializer="json",
        accept_content=["json"],
        result_serializer="json",
        timezone="UTC",
        enable_utc=True,
        task_track_started=True,
        imports=("workers.tasks", "api.worker"),
    )
    return app


celery_app = _make_celery()
