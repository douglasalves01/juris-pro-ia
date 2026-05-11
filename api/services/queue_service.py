"""Publicação de mensagens no RabbitMQ.

O FastAPI publica aqui quando recebe um job via HTTP.
O worker consome, processa e publica o resultado de volta.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

import aio_pika

logger = logging.getLogger(__name__)

RABBITMQ_URL = os.getenv("RABBITMQ_URL", "amqp://guest:guest@rabbitmq:5672//")

QUEUE_ANALYSIS = "jurispro.analysis"
QUEUE_KNOWLEDGE = "jurispro.knowledge"
QUEUE_MONITOR = "jurispro.monitor"


async def publish(queue: str, payload: dict[str, Any]) -> None:
    """Publica uma mensagem em uma fila RabbitMQ."""
    try:
        connection = await aio_pika.connect_robust(RABBITMQ_URL)
        async with connection:
            channel = await connection.channel()
            await channel.declare_queue(queue, durable=True)
            await channel.default_exchange.publish(
                aio_pika.Message(
                    body=json.dumps(payload, ensure_ascii=False, default=str).encode(),
                    content_type="application/json",
                    delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
                ),
                routing_key=queue,
            )
    except Exception as exc:
        logger.error("Falha ao publicar na fila %s: %s", queue, exc)
        raise


async def enqueue_analysis(
    job_id: str,
    text: str | None,
    tmp_path: str | None,
    regiao: str,
    mode: str,
    contract_id: str,
    firm_id: str | None,
) -> None:
    await publish(QUEUE_ANALYSIS, {
        "jobId": job_id,
        "text": text,
        "tmpPath": tmp_path,
        "regiao": regiao,
        "mode": mode,
        "contractId": contract_id,
        "firmId": firm_id,
    })


async def enqueue_knowledge(firm_id: str, documents: list[dict[str, Any]]) -> None:
    await publish(QUEUE_KNOWLEDGE, {"firmId": firm_id, "documents": documents})


async def enqueue_monitor(decisions: list[dict[str, Any]]) -> None:
    await publish(QUEUE_MONITOR, {"decisions": decisions})
