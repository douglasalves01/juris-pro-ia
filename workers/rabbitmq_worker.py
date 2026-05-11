"""Worker RabbitMQ — consome jobs de análise publicados pelo NestJS.

Fluxo:
  NestJS → publica em "jurispro.analysis" (exchange: direct)
  Worker → consome, processa com o pipeline de IA
  Worker → publica resultado em "jurispro.results" (exchange: direct)
  NestJS → consome o resultado e persiste/notifica o cliente

Execução:
  python -m workers.rabbitmq_worker

Variáveis de ambiente:
  RABBITMQ_URL  — ex: amqp://guest:guest@rabbitmq:5672//
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import aio_pika
import aio_pika.abc

logger = logging.getLogger(__name__)

RABBITMQ_URL = os.getenv("RABBITMQ_URL", "amqp://guest:guest@rabbitmq:5672//")

# Filas
QUEUE_ANALYSIS = "jurispro.analysis"       # NestJS → Worker (jobs de análise)
QUEUE_KNOWLEDGE = "jurispro.knowledge"     # NestJS → Worker (ingestão RAG)
QUEUE_MONITOR = "jurispro.monitor"         # NestJS → Worker (scan de jurisprudência)
QUEUE_RESULTS = "jurispro.results"         # Worker → NestJS (resultados)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


async def _publish_result(
    channel: aio_pika.abc.AbstractChannel,
    result: dict,
) -> None:
    """Publica o resultado na fila de retorno para o NestJS consumir."""
    await channel.default_exchange.publish(
        aio_pika.Message(
            body=json.dumps(result, ensure_ascii=False, default=str).encode(),
            content_type="application/json",
            delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
        ),
        routing_key=QUEUE_RESULTS,
    )


async def _handle_analysis(
    message: aio_pika.abc.AbstractIncomingMessage,
    channel: aio_pika.abc.AbstractChannel,
) -> None:
    """Processa um job de análise de documento."""
    async with message.process(requeue_on_error=True):
        body = json.loads(message.body.decode())
        job_id = str(body.get("jobId") or uuid.uuid4())
        contract_id = str(body.get("contractId") or "")
        regiao = str(body.get("regiao") or "SP")
        mode = str(body.get("mode") or "standard")
        firm_id = body.get("firmId")
        # Texto direto ou caminho de arquivo temporário
        text: str | None = body.get("text")
        tmp_path: str | None = body.get("tmpPath")

        started_at = _utc_now()
        t0 = time.perf_counter()

        logger.info("Processando job %s (mode=%s)", job_id, mode)

        from api.ml.pipeline import AnalysisPipeline, AnalysisResult
        from api.ml.text_extractor import TextExtractor
        from api.main import (
            _build_analysis_response,
            _classify_pipeline_failure,
            _error_payload,
            _http_error_contract,
            _maybe_debug_detail,
        )

        pipeline = AnalysisPipeline()
        extractor = TextExtractor()

        # Extração de texto se veio arquivo
        if not text and tmp_path:
            try:
                text = extractor.extract(tmp_path)
            except Exception as exc:
                logger.warning("Job %s: falha na extração: %s", job_id, exc)
                code, retryable, msg = _http_error_contract(422, str(exc))
                result = _error_payload(
                    job_id=job_id, contract_id=contract_id,
                    code=code, message=msg, retryable=retryable,
                    detail=_maybe_debug_detail(exc),
                    started_at=started_at, finished_at=_utc_now(),
                    duration_ms=int((time.perf_counter() - t0) * 1000),
                    mode=mode, trace_step="extract_text",
                )
                await _publish_result(channel, result)
                return
            finally:
                if tmp_path:
                    Path(tmp_path).unlink(missing_ok=True)

        if not text or not text.strip():
            code, retryable, msg = _http_error_contract(422, "Nenhum texto extraído.")
            result = _error_payload(
                job_id=job_id, contract_id=contract_id,
                code=code, message=msg, retryable=retryable,
                detail=None, started_at=started_at, finished_at=_utc_now(),
                duration_ms=int((time.perf_counter() - t0) * 1000),
                mode=mode, trace_step="extract_text",
            )
            await _publish_result(channel, result)
            return

        # Pipeline de IA
        try:
            analysis: AnalysisResult = pipeline.analyze(
                text,
                regiao=regiao,
                mode=mode,
                firm_id=uuid.UUID(str(firm_id)) if firm_id else None,
            )
        except Exception as exc:
            logger.exception("Job %s: falha no pipeline", job_id)
            code, retryable, msg = _classify_pipeline_failure(exc)
            result = _error_payload(
                job_id=job_id, contract_id=contract_id,
                code=code, message=msg, retryable=retryable,
                detail=_maybe_debug_detail(exc),
                started_at=started_at, finished_at=_utc_now(),
                duration_ms=int((time.perf_counter() - t0) * 1000),
                mode=mode, trace_step="pipeline",
            )
            await _publish_result(channel, result)
            return

        finished_at = _utc_now()
        result = _build_analysis_response(
            result=analysis,
            job_id=job_id,
            contract_id=contract_id,
            started_at=started_at,
            finished_at=finished_at,
            duration_ms=int((time.perf_counter() - t0) * 1000),
            mode=mode,
        )
        await _publish_result(channel, result)
        logger.info("Job %s concluído em %.1fs", job_id, time.perf_counter() - t0)


async def _handle_knowledge(
    message: aio_pika.abc.AbstractIncomingMessage,
    channel: aio_pika.abc.AbstractChannel,
) -> None:
    """Processa ingestão de documentos no RAG privado do escritório."""
    async with message.process(requeue_on_error=True):
        body = json.loads(message.body.decode())
        firm_id = str(body.get("firmId") or "")
        documents = body.get("documents") or []
        from api.services.private_knowledge import ingest_documents
        count = ingest_documents(firm_id, documents)
        logger.info("RAG: %d documentos indexados para firm %s", count, firm_id)


async def _handle_monitor(
    message: aio_pika.abc.AbstractIncomingMessage,
    channel: aio_pika.abc.AbstractChannel,
) -> None:
    """Processa scan de novas decisões jurisprudenciais."""
    async with message.process(requeue_on_error=True):
        body = json.loads(message.body.decode())
        decisions = body.get("decisions") or []
        from api.main import process_monitor_decisions
        count = len(process_monitor_decisions(decisions))
        logger.info("Monitor: %d alertas gerados", count)


async def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger.info("Conectando ao RabbitMQ: %s", RABBITMQ_URL.split("@")[-1])

    connection = await aio_pika.connect_robust(RABBITMQ_URL)
    async with connection:
        channel = await connection.channel()
        await channel.set_qos(prefetch_count=1)  # processa um job por vez por worker

        # Declara filas duráveis
        q_analysis = await channel.declare_queue(QUEUE_ANALYSIS, durable=True)
        q_knowledge = await channel.declare_queue(QUEUE_KNOWLEDGE, durable=True)
        q_monitor = await channel.declare_queue(QUEUE_MONITOR, durable=True)
        await channel.declare_queue(QUEUE_RESULTS, durable=True)

        await q_analysis.consume(lambda msg: _handle_analysis(msg, channel))
        await q_knowledge.consume(lambda msg: _handle_knowledge(msg, channel))
        await q_monitor.consume(lambda msg: _handle_monitor(msg, channel))

        logger.info("Worker aguardando mensagens. Ctrl+C para parar.")
        await asyncio.Future()  # roda indefinidamente


if __name__ == "__main__":
    asyncio.run(main())
