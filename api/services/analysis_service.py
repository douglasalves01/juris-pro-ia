"""Fila Celery e leitura de status de task."""

from __future__ import annotations

import uuid

from celery.result import AsyncResult

from workers.celery_app import celery_app


def enqueue_process_document(document_id: uuid.UUID) -> str:
    from workers.tasks import process_document

    async_result = process_document.delay(str(document_id))
    return async_result.id


def get_task_progress(task_id: str) -> tuple[str, int, str | None]:
    """Retorna (status_domínio, progresso 0-100, detalhe opcional)."""
    res = AsyncResult(task_id, app=celery_app)
    state = res.state
    meta = res.info if isinstance(res.info, dict) else {}
    progress = int(meta.get("progress", 0)) if isinstance(meta, dict) else 0
    detail: str | None = None
    if isinstance(meta, dict) and "detail" in meta:
        detail = str(meta["detail"])

    if state == "PENDING":
        return "pending", 0, detail
    if state in ("STARTED", "RETRY"):
        return "processing", max(progress, 5), detail
    if state == "PROGRESS":
        return "processing", max(progress, 10), detail
    if state == "SUCCESS":
        return "done", 100, detail
    if state == "FAILURE":
        err = str(res.result) if res.result else "falha"
        return "error", 0, err[:500]
    return "processing", progress, detail
