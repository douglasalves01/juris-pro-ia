"""Worker RabbitMQ — consome jobs de análise publicados pelo backend NestJS.

Fluxo:
  NestJS → publica em fila "jurispro.jobs.analysis.requests"
  Worker → consome, processa com o pipeline de IA
  Worker → publica resultado no exchange "jurispro.jobs" com routing key "analysis.results"
  NestJS → consome o resultado e persiste/notifica o cliente

Execução:
  python -m workers.rabbitmq_worker
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
from typing import Any

import aio_pika
import aio_pika.abc

logger = logging.getLogger(__name__)

RABBITMQ_URL = os.getenv("RABBITMQ_URL", "amqp://guest:guest@rabbitmq:5672//")

# Fila que o NestJS publica (a gente consome)
QUEUE_REQUESTS = "jurispro.jobs.analysis.requests"

# Exchange e routing key para publicar resultado (NestJS consome)
EXCHANGE_JOBS = "jurispro.jobs"
ROUTING_KEY_RESULTS = "analysis.results"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _analysis_result_log_payload(payload: dict[str, Any]) -> dict[str, Any]:
    result = payload.get("result") if isinstance(payload.get("result"), dict) else {}
    trace = result.get("aiTrace") if isinstance(result.get("aiTrace"), dict) else {}
    return {
        "jobId": payload.get("jobId"),
        "firmId": payload.get("firmId"),
        "contractId": payload.get("contractId"),
        "success": payload.get("success"),
        "risk": {
            "score": result.get("riskScore"),
            "level": result.get("riskLevel"),
            "mainRisks": result.get("mainRisks"),
        },
        "outcome": {
            "probability": result.get("outcomeProb"),
            "rationale": result.get("outcomeRationale"),
            "confidence": result.get("outcomeConfidence"),
        },
        "recommendations": result.get("recommendations"),
        "positivePoints": result.get("positivePoints"),
        "counts": {
            "attentionPoints": len(result.get("attentionPoints") or []),
            "entities": len(result.get("entities") or []),
            "similarCases": len(result.get("similarCases") or []),
        },
        "trace": {
            "provider": trace.get("provider"),
            "model": trace.get("model"),
            "latencyMs": trace.get("latencyMs"),
            "steps": [step.get("step") for step in (trace.get("steps") or []) if isinstance(step, dict)],
        },
    }


def _format_success_result(
    job_id: str,
    firm_id: str | None,
    contract_id: str,
    analysis: Any,
    duration_ms: int,
) -> dict[str, Any]:
    """Formata o resultado no formato que o backend espera."""
    # Attention points
    attention_points = []
    for ap in (analysis.attention_points or []):
        attention_points.append({
            "severity": ap.severidade,
            "clause": ap.clausula_referencia,
            "description": ap.descricao,
            "page": None,
        })

    # Entities
    entities = []
    ent = analysis.entities
    for name in (ent.pessoas or []):
        entities.append({"type": "PARTE", "value": name, "confidence": 0.85})
    for org in (ent.organizacoes or []):
        entities.append({"type": "ORGANIZACAO", "value": org, "confidence": 0.85})
    for leg in (ent.legislacao or []):
        entities.append({"type": "LEGISLACAO", "value": leg, "confidence": 0.80})
    for data in (ent.datas or []):
        entities.append({"type": "DATA", "value": data, "confidence": 0.80})
    for valor in (ent.valores or []):
        entities.append({"type": "VALOR", "value": valor, "confidence": 0.80})

    # Similar cases
    similar_cases = []
    for sc in (analysis.similar_cases or []):
        similar_cases.append({
            "caseId": sc.id if sc.id else None,
            "tribunal": sc.tribunal,
            "number": sc.number,
            "similarity": sc.similaridade,
            "outcome": sc.outcome,
            "summary": sc.resumo,
        })

    # AI Trace
    from api.ml.pipeline import AnalysisPipeline
    pipeline = AnalysisPipeline()
    ext_trace = pipeline.last_external_trace
    steps = []
    for step in pipeline.last_steps:
        steps.append({
            "step": step.get("step", "unknown"),
            "provider": step.get("provider", "internal"),
            "model": step.get("model"),
            "latencyMs": step.get("durationMs", 0),
        })

    ai_trace = {
        "pipelineVersion": "3.0.0",
        "provider": ext_trace.get("provider") or "internal",
        "model": ext_trace.get("model") or "jurispro-local",
        "costCents": int((ext_trace.get("cost_usd") or 0) * 100),
        "latencyMs": duration_ms,
        "steps": steps,
    }

    return {
        "jobId": job_id,
        "firmId": firm_id or "",
        "contractId": contract_id,
        "success": True,
        "result": {
            "riskScore": analysis.risk_score,
            "riskLevel": analysis.risk_level.upper(),
            "summary": analysis.executive_summary,
            "mainRisks": analysis.main_risks,
            "positivePoints": analysis.positive_points,
            "recommendations": analysis.recommendations,
            "feeMin": analysis.fee_estimate_min,
            "feeSuggested": analysis.fee_estimate_suggested,
            "feeMax": analysis.fee_estimate_max,
            "outcomeProb": analysis.win_probability,
            "outcomeRationale": analysis.win_prediction,
            "outcomeConfidence": analysis.win_confidence,
            "attentionPoints": attention_points,
            "entities": entities,
            "similarCases": similar_cases,
            "aiTrace": ai_trace,
        },
    }


def _format_error_result(
    job_id: str,
    firm_id: str | None,
    contract_id: str,
    error_message: str,
) -> dict[str, Any]:
    """Formata erro no formato que o backend espera (error como string)."""
    return {
        "jobId": job_id,
        "firmId": firm_id or "",
        "contractId": contract_id,
        "success": False,
        "error": error_message,
    }


async def _publish_result(
    channel: aio_pika.abc.AbstractChannel,
    exchange: aio_pika.abc.AbstractExchange,
    result: dict,
) -> None:
    """Publica o resultado no exchange jurispro.jobs com routing key analysis.results."""
    await exchange.publish(
        aio_pika.Message(
            body=json.dumps(result, ensure_ascii=False, default=str).encode(),
            content_type="application/json",
            delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
        ),
        routing_key=ROUTING_KEY_RESULTS,
    )


async def _handle_analysis(
    message: aio_pika.abc.AbstractIncomingMessage,
    channel: aio_pika.abc.AbstractChannel,
    exchange: aio_pika.abc.AbstractExchange,
) -> None:
    """Processa um job de análise de documento."""
    async with message.process():
        body = json.loads(message.body.decode())
        job_id = str(body.get("jobId") or body.get("id") or uuid.uuid4())
        contract_id = str(body.get("contractId") or "")
        regiao = str(body.get("regiao") or "SP")
        mode = str(body.get("mode") or "standard")
        firm_id = body.get("firmId")
        text: str | None = body.get("text")
        file_path: str | None = body.get("filePath")

        # O backend manda filePath relativo ou absoluto
        # No nosso container resolve para /app/storage/contracts/{filePath}
        if file_path:
            if file_path.startswith("/app/") and not Path(file_path).exists():
                relative = file_path[len("/app/"):]
                file_path = f"/app/storage/contracts/{relative}"
            elif not file_path.startswith("/"):
                file_path = f"/app/storage/contracts/{file_path}"

        started_at = _utc_now()
        t0 = time.perf_counter()

        logger.info("Processando job %s (mode=%s)", job_id, mode)

        from api.ml.pipeline import AnalysisPipeline, AnalysisResult
        from api.ml.text_extractor import TextExtractor

        pipeline = AnalysisPipeline()
        extractor = TextExtractor()

        # Extração de texto se veio caminho de arquivo
        if not text and file_path:
            try:
                text = extractor.extract(file_path)
            except Exception as exc:
                logger.warning("Job %s: falha na extração: %s", job_id, exc)
                result = _format_error_result(job_id, firm_id, contract_id, str(exc))
                await _publish_result(channel, exchange, result)
                return

        if not text or not text.strip():
            result = _format_error_result(
                job_id, firm_id, contract_id,
                "Nenhum texto extraído do arquivo.",
            )
            await _publish_result(channel, exchange, result)
            return

        # Pipeline de IA
        try:
            firm_uuid = None
            if firm_id:
                try:
                    firm_uuid = uuid.UUID(str(firm_id))
                except ValueError:
                    firm_uuid = None

            analysis: AnalysisResult = pipeline.analyze(
                text,
                regiao=regiao,
                mode=mode,
                firm_id=firm_uuid,
            )
        except Exception as exc:
            logger.exception("Job %s: falha no pipeline", job_id)
            result = _format_error_result(job_id, firm_id, contract_id, str(exc))
            await _publish_result(channel, exchange, result)
            return

        duration_ms = int((time.perf_counter() - t0) * 1000)

        result = _format_success_result(
            job_id=job_id,
            firm_id=firm_id,
            contract_id=contract_id,
            analysis=analysis,
            duration_ms=duration_ms,
        )
        logger.info(
            "analysis_result_to_backend=%s",
            json.dumps(_analysis_result_log_payload(result), ensure_ascii=False),
        )
        await _publish_result(channel, exchange, result)
        logger.info("Job %s concluído em %.1fs", job_id, time.perf_counter() - t0)


async def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger.info("Conectando ao RabbitMQ: %s", RABBITMQ_URL.split("@")[-1])

    connection = await aio_pika.connect_robust(RABBITMQ_URL)
    async with connection:
        channel = await connection.channel()
        await channel.set_qos(prefetch_count=1)

        # Declara o exchange (direct) — mesmo que o NestJS já tenha criado
        exchange = await channel.declare_exchange(
            EXCHANGE_JOBS, aio_pika.ExchangeType.DIRECT, durable=True
        )

        # Declara a fila e faz bind no exchange
        queue = await channel.declare_queue(QUEUE_REQUESTS, durable=True)
        await queue.bind(exchange, routing_key="analysis.requests")

        # Consome
        await queue.consume(lambda msg: _handle_analysis(msg, channel, exchange))

        logger.info("Worker escutando fila '%s'. Ctrl+C para parar.", QUEUE_REQUESTS)
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
