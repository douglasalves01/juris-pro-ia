"""JurisPro IA — microserviço FastAPI (contrato de análise unificado)."""

from __future__ import annotations

import asyncio
import collections
import copy
import hashlib
import json
import logging
import math
import os
import tempfile
import threading
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import BackgroundTasks, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ValidationError

from api.config import get_settings
from api.ml.pipeline import AnalysisPipeline, AnalysisResult
from api.ml.models.obligation_extractor import extract as extract_obligations
from api.ml.models.quality_scorer import score as score_quality
from api.ml.models.urgency_classifier import classify as classify_urgency
from api.ml.preprocessor import TextPreprocessor
from api.ml.models.contract_differ import CompareResult
from api.schemas.analysis import (
    AnalysisResponse,
    CounterArgumentsRequest,
    CounterArgumentsResponse,
    DraftRequest,
    DraftResponse,
    FirmKnowledgeIngestRequest,
    FirmKnowledgeIngestResponse,
    FirmKnowledgeStatsResponse,
    MonitorAlertsResponse,
    MonitorSubscribeRequest,
    MonitorSubscribeResponse,
    ObligationsRequest,
    ObligationsResponse,
    PipelineMetricsResponse,
    PlainSummaryRequest,
    PlainSummaryResponse,
    QualityRequest,
    QualityResponse,
    UrgencyRequest,
    UrgencyResponse,
)
from api.services.counter_arguments_service import generate_counter_arguments
from api.services.draft_generation_service import generate_draft
from api.services.jurisprudence_monitor import build_alerts, notify_alert
from api.services.obligations_webhook import notify_obligations
from api.services.plain_summary_service import generate_pdf_base64, generate_summary
from api.services.private_knowledge import ingest_documents as ingest_firm_knowledge
from api.services.private_knowledge import stats as firm_knowledge_stats
from api.services.semantic_cache import stats as semantic_cache_stats
from api.services.auth_service import verify_access_token
from api.ml.text_extractor import TextExtractor

_MODELS_DIR = os.getenv("MODELS_DIR", str(Path(__file__).resolve().parent.parent / "hf_models"))
_MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(50 * 1024 * 1024)))
_CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")

_ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt", ".text"}
_CACHE_TTL_SECONDS = 3600
_CACHE_MAX_ENTRIES = 200
_CACHE_CLEANUP_INTERVAL = 100
_MAX_ANALYSIS_JOBS = int(os.getenv("JURISPRO_MAX_ASYNC_JOBS", "500"))
_METRICS_BUFFER_SIZE = 100
_PIPELINE_VERSION = "3.0.0"
_DEBUG_ERRORS = os.getenv("JURISPRO_DEBUG_ERRORS", "").strip().lower() in ("1", "true", "yes")
_metrics_buffer: collections.deque[list[dict[str, Any]]] = collections.deque(
    maxlen=_METRICS_BUFFER_SIZE
)
_metrics_lock = threading.Lock()

logger = logging.getLogger(__name__)


def _log_async_task_result(task: asyncio.Task) -> None:
    try:
        exc = task.exception()
        if exc:
            logger.error("Tarefa de análise assíncrona encerrou com erro: %s", exc)
    except asyncio.CancelledError:
        pass


@asynccontextmanager
async def lifespan(app: FastAPI):
    pipeline = AnalysisPipeline(_MODELS_DIR)
    app.state.pipeline = pipeline
    app.state.extractor = TextExtractor()
    app.state.cache = {}
    app.state.cache_lock = threading.Lock()
    app.state.cache_requests = 0
    app.state.analysis_jobs = {}
    app.state.jobs_lock = threading.Lock()
    app.state.monitor_subscriptions = {}
    app.state.monitor_alerts = {}
    app.state.monitor_lock = threading.Lock()
    yield
    app.state.pipeline = None
    app.state.extractor = None
    app.state.cache = {}
    app.state.analysis_jobs = {}
    app.state.monitor_subscriptions = {}
    app.state.monitor_alerts = {}


app = FastAPI(
    title="JurisPro IA",
    version="3.0.0",
    description="Análise jurídica com modelos locais; um único contrato JSON por endpoint.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AnalyzeTextBody(BaseModel):
    text: str
    regiao: str = "SP"
    mode: str = "standard"
    jobId: str | None = None
    contractId: str | None = None
    firmId: str | None = None


def _cache_key(
    text: str, regiao: str, mode: str = "standard", *, source: str = "file"
) -> str:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return f"{source}:{mode.upper()}:{regiao.upper()}:{digest}"


def _sse_event(event: str, data: dict[str, Any]) -> str:
    payload = json.dumps(data, ensure_ascii=False, default=str)
    return f"event: {event}\ndata: {payload}\n\n"


def _authorize_firm_request(firm_id: str, authorization: str | None) -> None:
    try:
        requested = uuid.UUID(str(firm_id))
    except ValueError:
        raise HTTPException(status_code=422, detail="firmId inválido.")
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Token Bearer obrigatório.")
    token = authorization.split(" ", 1)[1].strip()
    try:
        claims = verify_access_token(token)
        token_firm = uuid.UUID(str(claims.get("firm_id")))
    except Exception:
        raise HTTPException(status_code=401, detail="Token inválido ou expirado.")
    if token_firm != requested:
        raise HTTPException(status_code=403, detail="firmId não autorizado.")


def _authorized_firm_uuid(firm_id: str | None, authorization: str | None) -> uuid.UUID | None:
    if not firm_id:
        return None
    _authorize_firm_request(firm_id, authorization)
    return uuid.UUID(str(firm_id))


def _normalize_response_for_cache(payload: dict[str, Any]) -> dict[str, Any]:
    """Remove identificadores de pedido para reutilizar resultado em outro job."""
    p = copy.deepcopy(payload)
    p["jobId"] = ""
    p["contractId"] = ""
    return p


def _patch_cached_response(
    cached: dict[str, Any],
    *,
    job_id: str,
    contract_id: str,
    started_at: str,
    finished_at: str,
    duration_ms: int,
) -> dict[str, Any]:
    out = copy.deepcopy(cached)
    out["jobId"] = job_id
    out["contractId"] = contract_id
    trace = out.get("trace")
    if isinstance(trace, dict):
        trace["startedAt"] = started_at
        trace["finishedAt"] = finished_at
        trace["durationMs"] = duration_ms
    return out


def _get_cached_payload(key: str) -> dict[str, Any] | None:
    now = time.time()
    with app.state.cache_lock:
        app.state.cache_requests += 1
        if app.state.cache_requests % _CACHE_CLEANUP_INTERVAL == 0:
            _cleanup_expired_cache(now)

        entry = app.state.cache.get(key)
        if not entry:
            return None

        inserted_at, payload = entry
        if now - inserted_at >= _CACHE_TTL_SECONDS:
            app.state.cache.pop(key, None)
            return None

        app.state.cache[key] = (now, payload)
        return copy.deepcopy(payload)


def _store_cached_payload(key: str, payload: dict[str, Any]) -> None:
    now = time.time()
    stored = _normalize_response_for_cache(payload)
    with app.state.cache_lock:
        if len(app.state.cache) >= _CACHE_MAX_ENTRIES and key not in app.state.cache:
            oldest_key = min(app.state.cache, key=lambda k: app.state.cache[k][0])
            app.state.cache.pop(oldest_key, None)
        app.state.cache[key] = (now, stored)


def _cleanup_expired_cache(now: float) -> None:
    expired = [
        key
        for key, (inserted_at, _) in app.state.cache.items()
        if now - inserted_at >= _CACHE_TTL_SECONDS
    ]
    for key in expired:
        app.state.cache.pop(key, None)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _queue_backend() -> str:
    return os.getenv("JURISPRO_QUEUE_BACKEND", get_settings().queue_backend).strip().lower()


def record_pipeline_metrics(steps: list[dict[str, Any]]) -> None:
    clean_steps = []
    for step in steps:
        duration = step.get("durationMs")
        if not isinstance(duration, int) or duration < 0:
            continue
        clean_steps.append(
            {
                "step": str(step.get("step") or "unknown"),
                "durationMs": duration,
            }
        )
    with _metrics_lock:
        _metrics_buffer.append(clean_steps)


def compute_pipeline_metrics() -> list[dict[str, Any]]:
    with _metrics_lock:
        snapshot = list(_metrics_buffer)

    per_step: dict[str, list[int]] = {}
    for execution_steps in snapshot:
        for step in execution_steps:
            duration = step.get("durationMs")
            if not isinstance(duration, int) or duration < 0:
                continue
            per_step.setdefault(str(step.get("step") or "unknown"), []).append(duration)

    metrics = []
    for step_name in sorted(per_step):
        durations = sorted(per_step[step_name])
        call_count = len(durations)
        p95_index = max(0, math.ceil(call_count * 0.95) - 1)
        metrics.append(
            {
                "step": step_name,
                "avgDurationMs": round(sum(durations) / call_count, 2),
                "p95DurationMs": durations[p95_index],
                "callCount": call_count,
            }
        )
    return metrics


def _normalize_risk_level(level: str) -> str:
    normalized = (level or "").strip().lower()
    mapping = {
        "baixo": "BAIXO",
        "baixa": "BAIXO",
        "médio": "MEDIO",
        "medio": "MEDIO",
        "média": "MEDIO",
        "media": "MEDIO",
        "alto": "ALTO",
        "alta": "ALTO",
        "crítico": "CRITICO",
        "critico": "CRITICO",
        "crítica": "CRITICO",
        "critica": "CRITICO",
    }
    return mapping.get(normalized, "MEDIO")


def _normalize_severity(severity: str) -> str:
    normalized = (severity or "").strip().lower()
    mapping = {
        "baixa": "low",
        "baixo": "low",
        "low": "low",
        "média": "medium",
        "medio": "medium",
        "médio": "medium",
        "media": "medium",
        "medium": "medium",
        "alta": "high",
        "alto": "high",
        "high": "high",
        "crítica": "critical",
        "critico": "critical",
        "crítico": "critical",
        "critica": "critical",
        "critical": "critical",
    }
    return mapping.get(normalized, "medium")


def _build_trace(
    started_at: str,
    finished_at: str,
    duration_ms: int,
    mode: str,
    external_api_used: bool = False,
) -> dict[str, Any]:
    return {
        "pipelineVersion": _PIPELINE_VERSION,
        "startedAt": started_at,
        "finishedAt": finished_at,
        "durationMs": duration_ms,
        "mode": mode,
        "externalApiUsed": external_api_used,
        "externalProvider": None,
        "externalModel": None,
        "estimatedCostUsd": 0.0,
        "localModelCostEstimateUsd": 0.0,
        "steps": [],
    }


def _normalize_trace_steps(raw_steps: list[dict[str, Any]], duration_ms: int) -> list[dict[str, Any]]:
    if not raw_steps:
        return [
            {
                "step": "pipeline",
                "provider": "internal",
                "model": "jurispro-local-pipeline",
                "modelVersion": _PIPELINE_VERSION,
                "durationMs": duration_ms,
                "startedAt": None,
                "finishedAt": None,
                "confidence": None,
                "inputTokens": None,
                "outputTokens": None,
                "estimatedCostUsd": 0.0,
            }
        ]

    out = []
    for step in raw_steps:
        provider = str(step.get("provider") or "internal")
        if provider not in {"huggingface", "openai", "gemini", "internal", "rules"}:
            provider = "internal"
        confidence = step.get("confidence")
        out.append(
            {
                "step": str(step.get("step") or "pipeline"),
                "provider": provider,
                "model": step.get("model"),
                "modelVersion": step.get("modelVersion") or _PIPELINE_VERSION,
                "durationMs": step.get("durationMs"),
                "startedAt": step.get("startedAt"),
                "finishedAt": step.get("finishedAt"),
                "confidence": confidence if isinstance(confidence, (float, int)) else None,
                "inputTokens": step.get("inputTokens"),
                "outputTokens": step.get("outputTokens"),
                "estimatedCostUsd": step.get("estimatedCostUsd", 0.0),
            }
        )
    return out


def _error_trace_payload(
    started_at: str,
    finished_at: str,
    duration_ms: int,
    mode: str,
    step: str = "request_failed",
) -> dict[str, Any]:
    """Trace mínimo auditável: sempre inclui ao menos um passo (task 08)."""
    trace = _build_trace(started_at, finished_at, duration_ms, mode)
    trace["steps"] = _normalize_trace_steps(
        [
            {
                "step": step,
                "provider": "internal",
                "model": None,
                "modelVersion": _PIPELINE_VERSION,
                "durationMs": duration_ms,
                "confidence": None,
                "inputTokens": None,
                "outputTokens": None,
                "estimatedCostUsd": 0.0,
            }
        ],
        duration_ms,
    )
    return trace


def _maybe_debug_detail(exc_or_detail: Any) -> Any:
    if not _DEBUG_ERRORS:
        return None
    if isinstance(exc_or_detail, BaseException):
        return str(exc_or_detail)
    return exc_or_detail


def _classify_extraction_error(detail: Any) -> tuple[str, str]:
    """422 na extração: código + mensagem segura para o cliente."""
    text = str(detail) if detail is not None else ""
    low = text.lower()
    if "text' não pode ser vazio" in low or (
        "campo" in low and "text" in low and "vazio" in low
    ):
        return (
            "OUTPUT_VALIDATION_FAILED",
            "O corpo da requisição precisa incluir texto não vazio.",
        )
    if "nenhum texto extraído" in low:
        return (
            "TEXT_EXTRACTION_FAILED",
            "Não foi possível extrair texto utilizável deste arquivo.",
        )
    if "tesseract" in low or "ocr indisponível" in low:
        return (
            "OCR_FAILED",
            "A leitura do PDF por OCR falhou ou não está disponível neste servidor. "
            "Prefira PDF com texto selecionável ou verifique a instalação do Tesseract.",
        )
    if "falha ao extrair texto" in low:
        return (
            "TEXT_EXTRACTION_FAILED",
            "Não foi possível ler o conteúdo do arquivo.",
        )
    return "TEXT_EXTRACTION_FAILED", "Não foi possível processar o arquivo enviado."


def _http_error_contract(
    status_code: int, detail: Any
) -> tuple[str, bool, str]:
    """HTTPException do upload/extração → código, retryable, mensagem pública."""
    if status_code == 413:
        return (
            "DOCUMENT_TOO_LARGE",
            False,
            "O arquivo excede o tamanho máximo permitido.",
        )
    if status_code == 415:
        return (
            "UNSUPPORTED_FILE",
            False,
            "Formato de arquivo não suportado. Use PDF, DOCX ou TXT.",
        )
    if status_code == 422:
        code, msg = _classify_extraction_error(detail)
        return code, False, msg
    if status_code == 409:
        return (
            "JOB_ALREADY_ACTIVE",
            False,
            "Este identificador de job já está em fila ou em processamento.",
        )
    return "UNKNOWN", False, "Requisição inválida."


def _classify_pipeline_failure(exc: BaseException) -> tuple[str, bool, str]:
    if isinstance(exc, FileNotFoundError):
        return (
            "MODEL_UNAVAILABLE",
            False,
            "Recurso de modelo indisponível no servidor. Contate o suporte.",
        )
    if isinstance(exc, ValidationError):
        return (
            "OUTPUT_VALIDATION_FAILED",
            False,
            "A análise produziu um resultado inválido.",
        )
    low = str(exc).lower()
    if "out of memory" in low or "cuda out of memory" in low:
        return (
            "UNKNOWN",
            True,
            "Recursos insuficientes para processar o documento. Tente um arquivo menor.",
        )
    return (
        "UNKNOWN",
        True,
        "Não foi possível concluir a análise. Tente novamente em instantes.",
    )


def _error_payload(
    *,
    job_id: str,
    contract_id: str,
    code: str,
    message: str,
    retryable: bool,
    detail: Any = None,
    started_at: str,
    finished_at: str,
    duration_ms: int,
    mode: str,
    trace_step: str = "request_failed",
) -> dict[str, Any]:
    trace = _error_trace_payload(
        started_at, finished_at, duration_ms, mode, step=trace_step
    )
    return {
        "jobId": job_id,
        "contractId": contract_id,
        "status": "error",
        "error": {
            "code": code,
            "message": message,
            "retryable": retryable,
            "detail": detail,
        },
        "trace": trace,
    }


def _error_json_response(
    *,
    job_id: str,
    contract_id: str,
    status_code: int,
    code: str,
    message: str,
    retryable: bool,
    detail: Any = None,
    started_at: str,
    finished_at: str,
    duration_ms: int,
    mode: str,
    trace_step: str = "request_failed",
) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content=_error_payload(
            job_id=job_id,
            contract_id=contract_id,
            code=code,
            message=message,
            retryable=retryable,
            detail=detail,
            started_at=started_at,
            finished_at=finished_at,
            duration_ms=duration_ms,
            mode=mode,
            trace_step=trace_step,
        ),
    )


def _jobs_store() -> dict[str, dict[str, Any]]:
    return getattr(app.state, "analysis_jobs", {})


def _jobs_lock() -> threading.Lock:
    return app.state.jobs_lock


def _monitor_lock() -> threading.Lock:
    lock = getattr(app.state, "monitor_lock", None)
    if lock is None:
        app.state.monitor_lock = threading.Lock()
        lock = app.state.monitor_lock
    return lock


def _monitor_subscriptions() -> dict[str, dict[str, Any]]:
    store = getattr(app.state, "monitor_subscriptions", None)
    if store is None:
        app.state.monitor_subscriptions = {}
        store = app.state.monitor_subscriptions
    return store


def _monitor_alerts() -> dict[str, list[dict[str, Any]]]:
    store = getattr(app.state, "monitor_alerts", None)
    if store is None:
        app.state.monitor_alerts = {}
        store = app.state.monitor_alerts
    return store


def process_monitor_decisions(decisions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    with _monitor_lock():
        subscriptions = copy.deepcopy(_monitor_subscriptions())
    alerts = build_alerts(subscriptions, decisions)
    if not alerts:
        return []
    now = _utc_now()
    with _monitor_lock():
        alert_store = _monitor_alerts()
        live_subs = _monitor_subscriptions()
        for alert in alerts:
            alert["createdAt"] = now
            case_id = str(alert["caseId"])
            existing_ids = {item.get("newDecisionId") for item in alert_store.get(case_id, [])}
            if alert["newDecisionId"] in existing_ids:
                continue
            alert_store.setdefault(case_id, []).append(alert)
            notify_alert(live_subs.get(case_id, {}).get("webhookUrl"), alert)
    return alerts


def _summary_from_analysis_payload(payload: dict[str, Any]) -> str | None:
    result = payload.get("result")
    if not isinstance(result, dict):
        return None
    final_opinion = result.get("finalOpinion")
    if isinstance(final_opinion, dict):
        summary = final_opinion.get("executiveSummary")
        if isinstance(summary, str) and summary.strip():
            return summary
    document = result.get("document")
    if isinstance(document, dict):
        summary = document.get("summary")
        if isinstance(summary, str) and summary.strip():
            return summary
    return None


def _attention_points_from_analysis_payload(payload: dict[str, Any]) -> list[dict[str, Any]]:
    result = payload.get("result")
    if not isinstance(result, dict):
        return []
    points = result.get("attentionPoints")
    if not isinstance(points, list):
        return []
    return [point for point in points if isinstance(point, dict)]


def _resolve_summary_input(
    *,
    text: str | None,
    job_id: str | None,
    contract_id: str | None,
) -> tuple[str | None, str]:
    if text is not None and text.strip():
        return job_id, text.strip()
    if not job_id and not contract_id:
        raise HTTPException(
            status_code=422,
            detail="Informe text, jobId ou contractId para gerar o resumo.",
        )

    lock = _jobs_lock()
    with lock:
        records = copy.deepcopy(_jobs_store())

    if job_id:
        rec = records.get(job_id)
        if not rec:
            raise HTTPException(status_code=404, detail="Job não encontrado.")
        response = rec.get("response")
        if isinstance(response, dict):
            summary = _summary_from_analysis_payload(response)
            if summary:
                return job_id, summary
        raise HTTPException(status_code=404, detail="Resumo do job não encontrado.")

    candidates = [
        (jid, rec)
        for jid, rec in records.items()
        if isinstance(rec, dict)
        and isinstance(rec.get("response"), dict)
        and rec["response"].get("contractId") == contract_id
    ]
    if not candidates:
        raise HTTPException(status_code=404, detail="Contrato não encontrado.")
    selected_job_id, selected = max(
        candidates,
        key=lambda item: str(item[1].get("updated_at") or item[1].get("created_at") or ""),
    )
    summary = _summary_from_analysis_payload(selected.get("response", {}))
    if not summary:
        raise HTTPException(status_code=404, detail="Resumo do contrato não encontrado.")
    return selected_job_id, summary


def _resolve_counter_input(
    *,
    text: str | None,
    job_id: str | None,
    contract_id: str | None,
) -> tuple[str | None, str, list[dict[str, Any]]]:
    if text is not None and text.strip():
        return job_id, text.strip(), []
    if not job_id and not contract_id:
        raise HTTPException(
            status_code=422,
            detail="Informe text, jobId ou contractId para gerar contra-argumentos.",
        )

    lock = _jobs_lock()
    with lock:
        records = copy.deepcopy(_jobs_store())

    if job_id:
        rec = records.get(job_id)
        if not rec:
            raise HTTPException(status_code=404, detail="Job não encontrado.")
        response = rec.get("response")
        if isinstance(response, dict):
            source_text = _summary_from_analysis_payload(response) or ""
            return job_id, source_text, _attention_points_from_analysis_payload(response)
        raise HTTPException(status_code=404, detail="Resultado do job não encontrado.")

    candidates = [
        (jid, rec)
        for jid, rec in records.items()
        if isinstance(rec, dict)
        and isinstance(rec.get("response"), dict)
        and rec["response"].get("contractId") == contract_id
    ]
    if not candidates:
        raise HTTPException(status_code=404, detail="Contrato não encontrado.")
    selected_job_id, selected = max(
        candidates,
        key=lambda item: str(item[1].get("updated_at") or item[1].get("created_at") or ""),
    )
    response = selected.get("response", {})
    source_text = _summary_from_analysis_payload(response) if isinstance(response, dict) else None
    return selected_job_id, source_text or "", _attention_points_from_analysis_payload(response)


def _evict_oldest_job_if_needed() -> None:
    store = _jobs_store()
    if len(store) < _MAX_ANALYSIS_JOBS:
        return
    terminal = {"done", "error"}
    candidates = [
        jid for jid, rec in store.items() if rec.get("phase") in terminal
    ]
    if not candidates:
        candidates = list(store.keys())
    oldest = min(
        candidates,
        key=lambda jid: store[jid].get("created_at", ""),
    )
    store.pop(oldest, None)


def _job_queued_payload(
    job_id: str,
    contract_id: str,
    created_at: str,
    updated_at: str,
) -> dict[str, Any]:
    return {
        "jobId": job_id,
        "contractId": contract_id,
        "status": "queued",
        "createdAt": created_at,
        "updatedAt": updated_at,
    }


def _celery_job_response(job_id: str, rec: dict[str, Any]) -> dict[str, Any]:
    from celery.result import AsyncResult

    from workers.celery_app import celery_app

    task_id = str(rec.get("celery_task_id") or "")
    if not task_id:
        return rec["response"]

    result = AsyncResult(task_id, app=celery_app)
    state = result.state
    if state == "SUCCESS":
        payload = result.result
        if isinstance(payload, dict):
            rec["phase"] = payload.get("status", "done")
            rec["response"] = payload
            rec["updated_at"] = _utc_now()
            return payload
    if state == "FAILURE":
        updated_at = _utc_now()
        payload = {
            "jobId": job_id,
            "contractId": rec.get("contract_id", ""),
            "status": "error",
            "createdAt": rec.get("created_at", updated_at),
            "updatedAt": updated_at,
            "error": {"message": str(result.result or "Falha no worker Celery.")[:500]},
        }
        rec["phase"] = "error"
        rec["response"] = payload
        rec["updated_at"] = updated_at
        return payload

    meta = result.info if isinstance(result.info, dict) else {}
    phase = "processing" if state in {"STARTED", "PROGRESS", "RETRY"} else "queued"
    updated_at = _utc_now()
    payload = {
        "jobId": job_id,
        "contractId": rec.get("contract_id", ""),
        "status": phase,
        "createdAt": rec.get("created_at", updated_at),
        "updatedAt": updated_at,
    }
    if meta:
        payload["progress"] = int(meta.get("progress", 0) or 0)
        if meta.get("detail"):
            payload["detail"] = str(meta["detail"])
    rec["phase"] = phase
    rec["response"] = payload
    rec["updated_at"] = updated_at
    return payload


def _run_analysis_job_in_thread(
    job_id: str,
    tmp_path: str,
    regiao: str,
    mode: str,
    contract_id: str,
    wall_started: str,
    firm_id: str | None = None,
) -> None:
    store = _jobs_store()
    lock = _jobs_lock()
    mode = mode if mode in {"fast", "standard", "deep"} else "standard"
    created_at = wall_started

    def _touch_response_processing() -> None:
        with lock:
            if job_id not in store:
                return
            store[job_id]["phase"] = "processing"
            store[job_id]["updated_at"] = _utc_now()
            store[job_id]["response"] = {
                "jobId": job_id,
                "contractId": contract_id,
                "status": "processing",
                "createdAt": created_at,
                "updatedAt": store[job_id]["updated_at"],
            }

    _touch_response_processing()

    text: str | None = None
    try:
        text = app.state.extractor.extract(tmp_path)
    except Exception as exc:
        logger.warning("Job %s: falha na extração: %s", job_id, exc)
        t0 = time.perf_counter()
        code, retryable, message = _http_error_contract(422, str(exc))
        payload = _error_payload(
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
        with lock:
            if job_id in store:
                store[job_id]["phase"] = "error"
                store[job_id]["response"] = payload
                store[job_id]["updated_at"] = _utc_now()
        return
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    if not text or not text.strip():
        t0 = time.perf_counter()
        code, retryable, message = _http_error_contract(
            422, "Nenhum texto extraído do arquivo."
        )
        payload = _error_payload(
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
        with lock:
            if job_id in store:
                store[job_id]["phase"] = "error"
                store[job_id]["response"] = payload
                store[job_id]["updated_at"] = _utc_now()
        return

    started_pipeline = _utc_now()
    t_pipeline = time.perf_counter()
    try:
        result: AnalysisResult = app.state.pipeline.analyze(
            text,
            regiao=regiao,
            mode=mode,
            firm_id=uuid.UUID(firm_id) if firm_id else None,
        )
    except Exception as exc:
        logger.exception("Job %s: falha no pipeline", job_id)
        code, retryable, message = _classify_pipeline_failure(exc)
        payload = _error_payload(
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
        with lock:
            if job_id in store:
                store[job_id]["phase"] = "error"
                store[job_id]["response"] = payload
                store[job_id]["updated_at"] = _utc_now()
        return

    finished_at = _utc_now()
    duration_ms = int((time.perf_counter() - t_pipeline) * 1000)
    payload = _build_analysis_response(
        result=result,
        job_id=job_id,
        contract_id=contract_id,
        started_at=started_pipeline,
        finished_at=finished_at,
        duration_ms=duration_ms,
        mode=mode,
    )
    with lock:
        if job_id in store:
            store[job_id]["phase"] = "done"
            store[job_id]["response"] = payload
            store[job_id]["updated_at"] = _utc_now()


async def _enqueue_analysis_job(
    job_id: str,
    tmp_path: str,
    regiao: str,
    mode: str,
    contract_id: str,
    wall_started: str,
    firm_id: str | None = None,
) -> None:
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(
        None,
        _run_analysis_job_in_thread,
        job_id,
        tmp_path,
        regiao,
        mode,
        contract_id,
        wall_started,
        firm_id,
    )


def _build_analysis_response(
    *,
    result: AnalysisResult,
    job_id: str,
    contract_id: str,
    started_at: str,
    finished_at: str,
    duration_ms: int,
    mode: str,
    plain_text: str | None = None,
) -> dict[str, Any]:
    language = "pt-BR"
    page_count = None
    text_quality = 0.8
    if plain_text is not None:
        wc = len((plain_text or "").split())
        page_count = 1
        if wc <= 0:
            text_quality = 0.0
        elif wc < 80:
            text_quality = 0.55
        else:
            text_quality = 0.95
    else:
        extractor = getattr(app.state, "extractor", None)
        extractor_meta = getattr(extractor, "last_metadata", None)
        if extractor_meta is not None:
            language = (
                "pt-BR"
                if extractor_meta.language_code == "pt"
                else extractor_meta.language_code
            )
            page_count = extractor_meta.num_pages
            if extractor_meta.word_count <= 0:
                text_quality = 0.0
            elif extractor_meta.word_count < 80:
                text_quality = 0.55
            elif extractor_meta.has_ocr:
                text_quality = 0.8
            else:
                text_quality = 0.95

    attention_points = [
        {
            "severity": _normalize_severity(point.severidade),
            "clause": point.tipo,
            "title": point.tipo.replace("_", " ").strip().title() or None,
            "description": point.descricao,
            "recommendation": None,
            "page": None,
            "evidence": point.clausula_referencia,
            "confidence": 0.75,
            "source": "rules" if point.referencia_tipo == "clause_classifier" else "hybrid",
        }
        for point in result.attention_points
    ]

    entities = []
    for entity_type, values in {
        "pessoa": result.entities.pessoas,
        "organizacao": result.entities.organizacoes,
        "legislacao": result.entities.legislacao,
        "data": result.entities.datas,
        "valor": result.entities.valores,
    }.items():
        for value in values:
            entities.append(
                {
                    "type": entity_type,
                    "value": value,
                    "normalizedValue": value,
                    "page": None,
                    "confidence": 0.8,
                    "source": "hybrid",
                }
            )

    similar_cases = [
        {
            "caseId": item.id,
            "tribunal": item.tribunal,
            "number": item.number,
            "similarity": item.similaridade,
            "outcome": item.outcome,
            "summary": item.resumo,
            "relevanceReason": item.titulo,
        }
        for item in result.similar_cases
    ]

    risk_level = _normalize_risk_level(result.risk_level)
    risk_rationale = "; ".join(result.main_risks) if result.main_risks else result.risk_level
    pipeline = getattr(app.state, "pipeline", None)
    raw_steps = pipeline.last_steps if pipeline is not None else []
    ext_trace: dict[str, Any] = {}
    if pipeline is not None:
        raw_ext = getattr(pipeline, "last_external_trace", {})
        if isinstance(raw_ext, dict):
            ext_trace = raw_ext
    ext_used = bool(ext_trace.get("used"))
    trace = _build_trace(
        started_at, finished_at, duration_ms, mode, external_api_used=ext_used
    )
    if ext_used:
        prov = ext_trace.get("provider")
        if prov in ("openai", "gemini"):
            trace["externalProvider"] = prov
        elif prov:
            trace["externalProvider"] = "openai"
        mod = ext_trace.get("model")
        trace["externalModel"] = str(mod) if mod is not None else None
        trace["estimatedCostUsd"] = float(ext_trace.get("cost_usd") or 0.0)
    trace["steps"] = _normalize_trace_steps(raw_steps, duration_ms)
    record_pipeline_metrics(trace["steps"])

    return {
        "jobId": job_id,
        "contractId": contract_id,
        "status": "done",
        "result": {
            "document": {
                "type": result.document_kind,
                "legalArea": result.contract_type,
                "language": language,
                "pageCount": page_count,
                "textQuality": text_quality,
                "summary": result.executive_summary,
            },
            "risk": {
                "score": result.risk_score,
                "level": risk_level,
                "rationale": risk_rationale,
                "confidence": 0.75,
            },
            "attentionPoints": attention_points,
            "entities": entities,
            "similarCases": similar_cases,
            "fees": {
                "min": result.fee_estimate_min,
                "suggested": result.fee_estimate_suggested,
                "max": result.fee_estimate_max,
                "rationale": "Estimativa automática baseada em área jurídica, região e risco.",
            },
            "outcomeProbability": {
                "value": result.win_probability,
                "rationale": result.win_prediction,
                "confidence": result.win_confidence,
            },
            "urgency": {
                "score": result.urgency.score,
                "level": result.urgency.level,
                "rationale": result.urgency.rationale,
            },
            "compliance": [
                {
                    "regulation": block.regulation,
                    "items": [
                        {
                            "id": item.id,
                            "description": item.description,
                            "status": item.status,
                            "evidence": item.evidence,
                        }
                        for item in block.items
                    ],
                }
                for block in result.compliance
            ],
            "obligations": [
                {
                    "subject": item.subject,
                    "obligation": item.obligation,
                    "deadline": item.deadline,
                    "deadlineAbsolute": item.deadlineAbsolute,
                    "confidence": item.confidence,
                }
                for item in result.obligations
            ],
            "finalOpinion": {
                "title": f"Parecer preliminar - {result.contract_type}",
                "executiveSummary": result.executive_summary,
                "legalAnalysis": risk_rationale,
                "recommendations": result.recommendations,
                "limitations": _final_opinion_limitations(mode, external_used=ext_used),
            },
        },
        "trace": trace,
    }


def _final_opinion_limitations(mode: str, *, external_used: bool = False) -> list[str]:
    base = [
        "Análise automatizada para apoio jurídico; revisar evidências e documentos originais.",
        "Probabilidades e honorários são estimativas e não substituem avaliação profissional.",
    ]
    if external_used:
        base.append(
            "Trecho complementar gerado por modelo de linguagem externo; validar com fontes primárias e critério profissional."
        )
    if mode == "fast":
        base.append(
            "Modo rápido: sem busca de casos similares e com resumo apenas por extração estruturada (sem T5)."
        )
    elif mode == "deep":
        base.append(
            "Modo profundo: sumarização e busca de similares usam parâmetros estendidos (100% local). "
            "Quando um LLM externo estiver configurado, o parecer poderá ser enriquecido automaticamente."
        )
    return base


async def _extract_upload_text(file: UploadFile) -> str:
    ext = Path(file.filename or "").suffix.lower()
    if ext not in _ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=f"Formato não suportado: '{ext}'. Use PDF, DOCX, TXT ou TEXT.",
        )

    content = await file.read()
    if len(content) > _MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Arquivo excede o limite de {_MAX_UPLOAD_BYTES // (1024*1024)} MB.",
        )

    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        text: str = app.state.extractor.extract(tmp_path)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Falha ao extrair texto: {exc}") from exc
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    if not text or not text.strip():
        raise HTTPException(status_code=422, detail="Nenhum texto extraído do arquivo.")

    return text


@app.post("/analyze/file", response_model=AnalysisResponse)
async def analyze_file(
    file: UploadFile = File(...),
    regiao: str = Form(default="SP"),
    jobId: str | None = Form(default=None),
    contractId: str | None = Form(default=None),
    firmId: str | None = Form(default=None),
    userId: str | None = Form(default=None),
    mode: str = Form(default="standard"),
    authorization: str | None = Header(default=None),
):
    del userId
    job_id = jobId or str(uuid4())
    contract_id = contractId or ""
    mode = mode if mode in {"fast", "standard", "deep"} else "standard"
    firm_uuid = _authorized_firm_uuid(firmId, authorization)
    started_at = _utc_now()
    t0 = time.perf_counter()

    try:
        text = await _extract_upload_text(file)
    except HTTPException as exc:
        finished_at = _utc_now()
        duration_ms = int((time.perf_counter() - t0) * 1000)
        code, retryable, message = _http_error_contract(exc.status_code, exc.detail)
        return _error_json_response(
            job_id=job_id,
            contract_id=contract_id,
            status_code=exc.status_code,
            code=code,
            message=message,
            retryable=retryable,
            detail=_maybe_debug_detail(exc.detail),
            started_at=started_at,
            finished_at=finished_at,
            duration_ms=duration_ms,
            mode=mode,
            trace_step="extract_text",
        )

    cache_text = text if firm_uuid is None else f"{text}\nfirm:{firm_uuid}"
    cache_key = _cache_key(cache_text, regiao, mode, source="file")
    cached = _get_cached_payload(cache_key)
    if cached is not None:
        finished_at = _utc_now()
        duration_ms = int((time.perf_counter() - t0) * 1000)
        return _patch_cached_response(
            cached,
            job_id=job_id,
            contract_id=contract_id,
            started_at=started_at,
            finished_at=finished_at,
            duration_ms=duration_ms,
        )

    try:
        result: AnalysisResult = app.state.pipeline.analyze(
            text, regiao=regiao, mode=mode, firm_id=firm_uuid
        )
    except Exception as exc:
        logger.exception("Falha no pipeline de análise (jobId=%s)", job_id)
        finished_at = _utc_now()
        duration_ms = int((time.perf_counter() - t0) * 1000)
        code, retryable, message = _classify_pipeline_failure(exc)
        return _error_json_response(
            job_id=job_id,
            contract_id=contract_id,
            status_code=500,
            code=code,
            message=message,
            retryable=retryable,
            detail=_maybe_debug_detail(exc),
            started_at=started_at,
            finished_at=finished_at,
            duration_ms=duration_ms,
            mode=mode,
            trace_step="pipeline",
        )

    finished_at = _utc_now()
    duration_ms = int((time.perf_counter() - t0) * 1000)
    payload = _build_analysis_response(
        result=result,
        job_id=job_id,
        contract_id=contract_id,
        started_at=started_at,
        finished_at=finished_at,
        duration_ms=duration_ms,
        mode=mode,
    )
    _store_cached_payload(cache_key, payload)
    return payload


@app.post("/analyze/file/stream")
async def analyze_file_stream(
    file: UploadFile = File(...),
    regiao: str = Form(default="SP"),
    jobId: str | None = Form(default=None),
    contractId: str | None = Form(default=None),
    firmId: str | None = Form(default=None),
    userId: str | None = Form(default=None),
    mode: str = Form(default="standard"),
    authorization: str | None = Header(default=None),
):
    del userId
    job_id = jobId or str(uuid4())
    contract_id = contractId or ""
    mode = mode if mode in {"fast", "standard", "deep"} else "standard"
    firm_uuid = _authorized_firm_uuid(firmId, authorization)
    started_at = _utc_now()
    t0 = time.perf_counter()

    try:
        text = await _extract_upload_text(file)
    except HTTPException as exc:
        finished_at = _utc_now()
        duration_ms = int((time.perf_counter() - t0) * 1000)
        code, retryable, message = _http_error_contract(exc.status_code, exc.detail)
        return _error_json_response(
            job_id=job_id,
            contract_id=contract_id,
            status_code=exc.status_code,
            code=code,
            message=message,
            retryable=retryable,
            detail=_maybe_debug_detail(exc.detail),
            started_at=started_at,
            finished_at=finished_at,
            duration_ms=duration_ms,
            mode=mode,
            trace_step="extract_text",
        )

    cache_text = text if firm_uuid is None else f"{text}\nfirm:{firm_uuid}"
    cache_key = _cache_key(cache_text, regiao, mode, source="file")

    async def event_stream():
        yield _sse_event(
            "queued",
            {"jobId": job_id, "contractId": contract_id, "status": "queued"},
        )
        cached = _get_cached_payload(cache_key)
        if cached is not None:
            finished_at = _utc_now()
            duration_ms = int((time.perf_counter() - t0) * 1000)
            payload = _patch_cached_response(
                cached,
                job_id=job_id,
                contract_id=contract_id,
                started_at=started_at,
                finished_at=finished_at,
                duration_ms=duration_ms,
            )
            yield _sse_event("done", payload)
            return

        yield _sse_event(
            "processing",
            {"jobId": job_id, "contractId": contract_id, "status": "processing"},
        )
        analysis_task = asyncio.create_task(
            asyncio.to_thread(
                app.state.pipeline.analyze,
                text,
                regiao=regiao,
                mode=mode,
                firm_id=firm_uuid,
            )
        )
        emitted_steps = 0
        try:
            while not analysis_task.done():
                raw_steps = getattr(app.state.pipeline, "last_steps", [])
                for step in raw_steps[emitted_steps:]:
                    yield _sse_event("step", {"jobId": job_id, "step": step})
                emitted_steps = len(raw_steps)
                await asyncio.sleep(0.05)
            result = await analysis_task
        except Exception as exc:
            logger.exception("Falha no pipeline SSE (jobId=%s)", job_id)
            finished_at = _utc_now()
            duration_ms = int((time.perf_counter() - t0) * 1000)
            code, retryable, message = _classify_pipeline_failure(exc)
            payload = _error_payload(
                job_id=job_id,
                contract_id=contract_id,
                code=code,
                message=message,
                retryable=retryable,
                detail=_maybe_debug_detail(exc),
                started_at=started_at,
                finished_at=finished_at,
                duration_ms=duration_ms,
                mode=mode,
                trace_step="pipeline",
            )
            yield _sse_event("error", payload)
            return

        raw_steps = getattr(app.state.pipeline, "last_steps", [])
        for step in raw_steps[emitted_steps:]:
            yield _sse_event("step", {"jobId": job_id, "step": step})

        finished_at = _utc_now()
        duration_ms = int((time.perf_counter() - t0) * 1000)
        payload = _build_analysis_response(
            result=result,
            job_id=job_id,
            contract_id=contract_id,
            started_at=started_at,
            finished_at=finished_at,
            duration_ms=duration_ms,
            mode=mode,
        )
        _store_cached_payload(cache_key, payload)
        if emitted_steps == 0:
            for step in payload.get("trace", {}).get("steps", []):
                yield _sse_event("step", {"jobId": job_id, "step": step})
        yield _sse_event("done", payload)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/analyze/file/async")
async def analyze_file_async(
    file: UploadFile = File(...),
    regiao: str = Form(default="SP"),
    jobId: str | None = Form(default=None),
    contractId: str | None = Form(default=None),
    firmId: str | None = Form(default=None),
    userId: str | None = Form(default=None),
    mode: str = Form(default="standard"),
    authorization: str | None = Header(default=None),
):
    """Aceita o arquivo, retorna 202 e processa em background (fila em memória)."""
    del userId
    job_id = jobId or str(uuid4())
    contract_id = contractId or ""
    mode = mode if mode in {"fast", "standard", "deep"} else "standard"
    firm_uuid = _authorized_firm_uuid(firmId, authorization)
    wall_started = _utc_now()
    t0 = time.perf_counter()

    ext = Path(file.filename or "").suffix.lower()
    if ext not in _ALLOWED_EXTENSIONS:
        duration_ms = int((time.perf_counter() - t0) * 1000)
        code, retryable, message = _http_error_contract(415, "")
        return JSONResponse(
            status_code=415,
            content=_error_payload(
                job_id=job_id,
                contract_id=contract_id,
                code=code,
                message=message,
                retryable=retryable,
                detail=None,
                started_at=wall_started,
                finished_at=_utc_now(),
                duration_ms=duration_ms,
                mode=mode,
                trace_step="validate_upload",
            ),
        )
    content = await file.read()
    if len(content) > _MAX_UPLOAD_BYTES:
        duration_ms = int((time.perf_counter() - t0) * 1000)
        code, retryable, message = _http_error_contract(413, "")
        return JSONResponse(
            status_code=413,
            content=_error_payload(
                job_id=job_id,
                contract_id=contract_id,
                code=code,
                message=message,
                retryable=retryable,
                detail=None,
                started_at=wall_started,
                finished_at=_utc_now(),
                duration_ms=duration_ms,
                mode=mode,
                trace_step="validate_upload",
            ),
        )

    store = _jobs_store()
    lock = _jobs_lock()
    with lock:
        if job_id in store and store[job_id].get("phase") not in ("done", "error"):
            duration_ms = int((time.perf_counter() - t0) * 1000)
            code, retryable, message = _http_error_contract(409, "")
            return JSONResponse(
                status_code=409,
                content=_error_payload(
                    job_id=job_id,
                    contract_id=contract_id,
                    code=code,
                    message=message,
                    retryable=retryable,
                    detail=None,
                    started_at=wall_started,
                    finished_at=_utc_now(),
                    duration_ms=duration_ms,
                    mode=mode,
                    trace_step="validate_upload",
                ),
            )
        _evict_oldest_job_if_needed()

    fd, tmp_path = tempfile.mkstemp(suffix=ext)
    try:
        os.write(fd, content)
    finally:
        os.close(fd)

    now = _utc_now()
    with lock:
        store[job_id] = {
            "phase": "queued",
            "contract_id": contract_id,
            "created_at": now,
            "updated_at": now,
            "response": _job_queued_payload(job_id, contract_id, now, now),
        }

    if _queue_backend() in {"celery", "redis"}:
        from api.worker import process_analysis_job

        async_result = process_analysis_job.delay(
            job_id, tmp_path, regiao, mode, contract_id, now, str(firm_uuid) if firm_uuid else None
        )
        with lock:
            if job_id in store:
                store[job_id]["celery_task_id"] = async_result.id
                if async_result.ready() and isinstance(async_result.result, dict):
                    store[job_id]["phase"] = async_result.result.get("status", "done")
                    store[job_id]["response"] = async_result.result
                store[job_id]["updated_at"] = _utc_now()
        backend = "celery"
    else:
        task = asyncio.create_task(
            _enqueue_analysis_job(
                job_id, tmp_path, regiao, mode, contract_id, now, str(firm_uuid) if firm_uuid else None
            )
        )
        task.add_done_callback(_log_async_task_result)
        backend = "memory"

    return JSONResponse(
        status_code=202,
        content={
            "jobId": job_id,
            "contractId": contract_id,
            "status": "queued",
            "queueBackend": backend,
            "pollUrl": f"/jobs/{job_id}",
            "createdAt": now,
            "updatedAt": now,
        },
    )


@app.get("/jobs/{job_id}")
async def get_job_status(job_id: str):
    """Consulta status ou resultado completo do job assíncrono."""
    lock = _jobs_lock()
    with lock:
        rec = _jobs_store().get(job_id)
        if rec and rec.get("celery_task_id") and rec.get("phase") not in {"done", "error"}:
            return _celery_job_response(job_id, rec)
    if not rec:
        raise HTTPException(status_code=404, detail="Job não encontrado.")
    return rec["response"]


@app.get("/health")
async def health():
    loaded = getattr(app.state, "pipeline", None) is not None
    return {"status": "ok", "models_loaded": loaded}


@app.get("/metrics/pipeline", response_model=PipelineMetricsResponse)
async def pipeline_metrics():
    return {
        "steps": compute_pipeline_metrics(),
        "collectedAt": _utc_now(),
        "semanticCache": semantic_cache_stats(),
    }


@app.post("/analyze/summary/plain", response_model=PlainSummaryResponse)
async def analyze_plain_summary(body: PlainSummaryRequest):
    resolved_job_id, source_text = _resolve_summary_input(
        text=body.text,
        job_id=body.jobId,
        contract_id=body.contractId,
    )
    settings = get_settings()
    summary_text = generate_summary(
        source_text,
        body.level,
        settings.openai_api_key,
        str(settings.openai_base_url),
        str(settings.openai_model),
    )
    pdf_base64 = None
    if body.include_pdf:
        try:
            pdf_base64 = generate_pdf_base64(summary_text, settings.brand_name)
        except Exception as exc:
            return JSONResponse(
                status_code=500,
                content={
                    "code": "PDF_GENERATION_FAILED",
                    "message": "Falha ao gerar PDF do resumo.",
                    "detail": _maybe_debug_detail(exc),
                    "pdfBase64": None,
                },
            )
    return {
        "jobId": resolved_job_id,
        "summaryText": summary_text,
        "level": body.level,
        "generatedAt": _utc_now(),
        "pdfBase64": pdf_base64,
    }


@app.post("/analyze/counter-arguments", response_model=CounterArgumentsResponse)
async def analyze_counter_arguments(body: CounterArgumentsRequest):
    resolved_job_id, source_text, attention_points = _resolve_counter_input(
        text=body.text,
        job_id=body.jobId,
        contract_id=body.contractId,
    )
    settings = get_settings()
    arguments = generate_counter_arguments(
        source_text,
        attention_points,
        body.maxArguments,
        settings.openai_api_key,
        str(settings.openai_base_url),
        str(settings.openai_model),
    )
    return {
        "jobId": resolved_job_id,
        "arguments": arguments,
        "generatedAt": _utc_now(),
    }


@app.post("/analyze/urgency", response_model=UrgencyResponse)
async def analyze_urgency(body: UrgencyRequest):
    if not body.text or not body.text.strip():
        raise HTTPException(status_code=422, detail="Campo 'text' não pode ser vazio.")
    preprocessor = TextPreprocessor()
    cleaned = preprocessor.clean(body.text)
    sections = preprocessor.extract_sections(cleaned)
    result = classify_urgency(cleaned, sections.get("datas", []), "")
    return {
        "score": result.score,
        "level": result.level,
        "rationale": result.rationale,
        "generatedAt": _utc_now(),
    }


@app.post("/analyze/obligations", response_model=ObligationsResponse)
async def analyze_obligations(body: ObligationsRequest):
    if not body.text or not body.text.strip():
        raise HTTPException(status_code=422, detail="Campo 'text' não pode ser vazio.")
    preprocessor = TextPreprocessor()
    cleaned = preprocessor.clean(body.text)
    sections = preprocessor.extract_sections(cleaned)
    obligations = extract_obligations(cleaned, sections.get("datas", []))
    obligation_payload = [
        {
            "subject": item.subject,
            "obligation": item.obligation,
            "deadline": item.deadline,
            "deadlineAbsolute": item.deadlineAbsolute,
            "confidence": item.confidence,
        }
        for item in obligations
    ]
    settings = get_settings()
    notify_obligations(
        settings.obligations_webhook_url,
        job_id=None,
        contract_id=None,
        obligations=obligation_payload,
    )
    return {
        "obligations": obligation_payload,
        "generatedAt": _utc_now(),
    }


@app.post("/analyze/quality", response_model=QualityResponse)
async def analyze_quality(body: QualityRequest):
    if not body.text or not body.text.strip():
        raise HTTPException(status_code=422, detail="Campo 'text' não pode ser vazio.")
    result = score_quality(body.text)
    return {
        "score": result.score,
        "dimensions": result.dimensions,
        "suggestions": [
            {
                "dimension": item.dimension,
                "issue": item.issue,
                "fix": item.fix,
            }
            for item in result.suggestions
        ],
    }


@app.post("/monitor/subscribe", response_model=MonitorSubscribeResponse)
async def monitor_subscribe(body: MonitorSubscribeRequest):
    subscribed_at = _utc_now()
    reference_text = body.contractId or body.caseId
    with _monitor_lock():
        _monitor_subscriptions()[body.caseId] = {
            "caseId": body.caseId,
            "contractId": body.contractId,
            "referenceText": reference_text,
            "threshold": body.threshold,
            "webhookUrl": body.webhookUrl,
            "subscribedAt": subscribed_at,
        }
        _monitor_alerts().setdefault(body.caseId, [])
    return {
        "caseId": body.caseId,
        "contractId": body.contractId,
        "threshold": body.threshold,
        "subscribedAt": subscribed_at,
    }


@app.get("/monitor/alerts/{case_id}", response_model=MonitorAlertsResponse)
async def monitor_alerts(case_id: str):
    with _monitor_lock():
        alerts = copy.deepcopy(_monitor_alerts().get(case_id, []))
    return {
        "caseId": case_id,
        "alerts": alerts,
        "collectedAt": _utc_now(),
    }


@app.post("/generate/draft", response_model=DraftResponse)
async def generate_draft_endpoint(body: DraftRequest):
    settings = get_settings()
    result = generate_draft(
        body.documentType,
        body.context.model_dump(),
        body.style,
        settings.openai_api_key,
        str(settings.openai_base_url),
        str(settings.openai_model),
        body.firmId,
    )
    return result


@app.post(
    "/firms/{firm_id}/knowledge/ingest",
    response_model=FirmKnowledgeIngestResponse,
)
async def ingest_firm_knowledge_endpoint(
    firm_id: str,
    body: FirmKnowledgeIngestRequest,
    background_tasks: BackgroundTasks,
    authorization: str | None = Header(default=None),
):
    _authorize_firm_request(firm_id, authorization)
    job_id = str(uuid4())
    documents = [item.model_dump() for item in body.documents]
    settings = get_settings()
    if settings.queue_backend.strip().lower() == "celery":
        from api.worker import index_firm_knowledge

        index_firm_knowledge.delay(firm_id, documents)
    else:
        background_tasks.add_task(ingest_firm_knowledge, firm_id, documents)
    return {
        "jobId": job_id,
        "firmId": str(uuid.UUID(str(firm_id))),
        "status": "queued",
        "documentsReceived": len(documents),
    }


@app.get(
    "/firms/{firm_id}/knowledge/stats",
    response_model=FirmKnowledgeStatsResponse,
)
async def firm_knowledge_stats_endpoint(
    firm_id: str,
    authorization: str | None = Header(default=None),
):
    _authorize_firm_request(firm_id, authorization)
    return firm_knowledge_stats(firm_id)


@app.post("/analyze/text", response_model=AnalysisResponse)
async def analyze_text(
    body: AnalyzeTextBody,
    authorization: str | None = Header(default=None),
):
    job_id = body.jobId or str(uuid4())
    contract_id = body.contractId or ""
    mode = body.mode if body.mode in {"fast", "standard", "deep"} else "standard"
    firm_uuid = _authorized_firm_uuid(body.firmId, authorization)
    started_at = _utc_now()
    t0 = time.perf_counter()

    if not body.text or not body.text.strip():
        finished_at = _utc_now()
        duration_ms = int((time.perf_counter() - t0) * 1000)
        code, retryable, message = _http_error_contract(
            422, "Campo 'text' não pode ser vazio."
        )
        return _error_json_response(
            job_id=job_id,
            contract_id=contract_id,
            status_code=422,
            code=code,
            message=message,
            retryable=retryable,
            detail=None,
            started_at=started_at,
            finished_at=finished_at,
            duration_ms=duration_ms,
            mode=mode,
            trace_step="validate_body",
        )

    cache_text = body.text if firm_uuid is None else f"{body.text}\nfirm:{firm_uuid}"
    cache_key = _cache_key(cache_text, body.regiao, mode, source="text")
    cached = _get_cached_payload(cache_key)
    if cached is not None:
        finished_at = _utc_now()
        duration_ms = int((time.perf_counter() - t0) * 1000)
        return _patch_cached_response(
            cached,
            job_id=job_id,
            contract_id=contract_id,
            started_at=started_at,
            finished_at=finished_at,
            duration_ms=duration_ms,
        )

    try:
        result: AnalysisResult = app.state.pipeline.analyze(
            body.text, regiao=body.regiao, mode=mode, firm_id=firm_uuid
        )
    except Exception as exc:
        logger.exception("Falha no pipeline de análise (jobId=%s)", job_id)
        finished_at = _utc_now()
        duration_ms = int((time.perf_counter() - t0) * 1000)
        code, retryable, message = _classify_pipeline_failure(exc)
        return _error_json_response(
            job_id=job_id,
            contract_id=contract_id,
            status_code=500,
            code=code,
            message=message,
            retryable=retryable,
            detail=_maybe_debug_detail(exc),
            started_at=started_at,
            finished_at=finished_at,
            duration_ms=duration_ms,
            mode=mode,
            trace_step="pipeline",
        )

    finished_at = _utc_now()
    duration_ms = int((time.perf_counter() - t0) * 1000)
    payload = _build_analysis_response(
        result=result,
        job_id=job_id,
        contract_id=contract_id,
        started_at=started_at,
        finished_at=finished_at,
        duration_ms=duration_ms,
        mode=mode,
        plain_text=body.text,
    )
    _store_cached_payload(cache_key, payload)
    return payload


@app.post("/analyze/compare", response_model=CompareResult)
async def compare_files(
    file_a: UploadFile = File(...),
    file_b: UploadFile = File(...),
):
    text_a = await _extract_upload_text(file_a)
    text_b = await _extract_upload_text(file_b)

    try:
        return app.state.pipeline.compare(text_a, text_b)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Erro na comparação de contratos: {exc}") from exc
