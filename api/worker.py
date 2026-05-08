"""Worker Celery para jobs assíncronos do contrato HTTP de análise."""

from __future__ import annotations

import logging
import time
import uuid
from pathlib import Path
from typing import Any

from workers.celery_app import celery_app

logger = logging.getLogger(__name__)


def _update_progress(task, progress: int, detail: str) -> None:
    if getattr(task.app.conf, "task_always_eager", False):
        return
    task.update_state(state="PROGRESS", meta={"progress": progress, "detail": detail})


@celery_app.task(bind=True, name="api.worker.process_analysis_job")
def process_analysis_job(
    self,
    job_id: str,
    tmp_path: str,
    regiao: str,
    mode: str,
    contract_id: str,
    wall_started: str,
    firm_id: str | None = None,
) -> dict[str, Any]:
    from api.main import (
        _build_analysis_response,
        _classify_pipeline_failure,
        _error_payload,
        _http_error_contract,
        _maybe_debug_detail,
        _utc_now,
        app,
    )
    from api.ml.pipeline import AnalysisPipeline, AnalysisResult
    from api.ml.text_extractor import TextExtractor

    mode = mode if mode in {"fast", "standard", "deep"} else "standard"
    extractor = getattr(app.state, "extractor", None) or TextExtractor()
    pipeline = getattr(app.state, "pipeline", None) or AnalysisPipeline()
    app.state.extractor = extractor
    app.state.pipeline = pipeline

    _update_progress(self, 15, "Extraindo texto")
    text: str | None = None
    try:
        text = extractor.extract(tmp_path)
    except Exception as exc:
        logger.warning("Job %s: falha na extração no worker: %s", job_id, exc)
        t0 = time.perf_counter()
        code, retryable, message = _http_error_contract(422, str(exc))
        return _error_payload(
            job_id=job_id,
            contract_id=contract_id,
            code=code,
            message=message,
            retryable=retryable,
            detail=_maybe_debug_detail(exc),
            started_at=wall_started,
            finished_at=_utc_now(),
            duration_ms=int((time.perf_counter() - t0) * 1000),
            mode=mode,
            trace_step="extract_text",
        )
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    if not text or not text.strip():
        t0 = time.perf_counter()
        code, retryable, message = _http_error_contract(422, "Nenhum texto extraído do arquivo.")
        return _error_payload(
            job_id=job_id,
            contract_id=contract_id,
            code=code,
            message=message,
            retryable=retryable,
            detail=None,
            started_at=wall_started,
            finished_at=_utc_now(),
            duration_ms=int((time.perf_counter() - t0) * 1000),
            mode=mode,
            trace_step="extract_text",
        )

    _update_progress(self, 45, "Executando pipeline")
    started_pipeline = _utc_now()
    t_pipeline = time.perf_counter()
    try:
        result: AnalysisResult = pipeline.analyze(
            text,
            regiao=regiao,
            mode=mode,
            firm_id=uuid.UUID(firm_id) if firm_id else None,
        )
    except Exception as exc:
        logger.exception("Job %s: falha no pipeline no worker", job_id)
        code, retryable, message = _classify_pipeline_failure(exc)
        return _error_payload(
            job_id=job_id,
            contract_id=contract_id,
            code=code,
            message=message,
            retryable=retryable,
            detail=_maybe_debug_detail(exc),
            started_at=started_pipeline,
            finished_at=_utc_now(),
            duration_ms=int((time.perf_counter() - t_pipeline) * 1000),
            mode=mode,
            trace_step="pipeline",
        )

    _update_progress(self, 90, "Montando resposta")
    finished_at = _utc_now()
    return _build_analysis_response(
        result=result,
        job_id=job_id,
        contract_id=contract_id,
        started_at=started_pipeline,
        finished_at=finished_at,
        duration_ms=int((time.perf_counter() - t_pipeline) * 1000),
        mode=mode,
    )


@celery_app.task(name="api.worker.scan_monitor_alerts")
def scan_monitor_alerts(decisions: list[dict[str, Any]]) -> int:
    from api.main import process_monitor_decisions

    return len(process_monitor_decisions(decisions))


@celery_app.task(name="api.worker.index_firm_knowledge")
def index_firm_knowledge(firm_id: str, documents: list[dict[str, Any]]) -> int:
    from api.services.private_knowledge import ingest_documents

    return ingest_documents(firm_id, documents)
