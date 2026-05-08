"""Tasks Celery: processamento pesado de documentos."""

from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import datetime, timezone

from sqlalchemy import select

from api.database import async_session_factory
from api.models.document import Analysis, AnalysisStatus, Document, DocumentStatus
from api.models.user import Firm
from api.services.document_service import extract_text_from_file

from workers.celery_app import celery_app

logger = logging.getLogger(__name__)


def _get_pipeline():
    from api.config import get_settings
    from api.ml.pipeline import AnalysisPipeline

    return AnalysisPipeline(str(get_settings().models_dir))


async def _process_document_async(task, document_id: str) -> None:
    doc_uuid = uuid.UUID(document_id)
    file_path: str | None = None
    regiao = ""
    async with async_session_factory() as session:
        doc = await session.get(Document, doc_uuid)
        if doc is None:
            logger.error("Documento %s não encontrado", document_id)
            raise ValueError(f"Documento {document_id} não encontrado")

        firm = await session.get(Firm, doc.firm_id)
        regiao = firm.region if firm else ""
        file_path = doc.file_path

        r = await session.execute(
            select(Analysis).where(Analysis.document_id == doc_uuid)
        )
        analysis = r.scalar_one_or_none()
        if analysis is None:
            analysis = Analysis(
                id=uuid.uuid4(),
                document_id=doc_uuid,
                status=AnalysisStatus.processing,
            )
            session.add(analysis)
        else:
            analysis.status = AnalysisStatus.processing
            analysis.error_msg = None
            analysis.completed_at = None

        doc.status = DocumentStatus.processing
        await session.commit()

    assert file_path is not None

    task.update_state(
        state="PROGRESS",
        meta={"progress": 15, "detail": "Extraindo texto do arquivo"},
    )

    try:
        text = extract_text_from_file(file_path)
    except Exception as exc:
        logger.exception("Falha na extração: %s", exc)
        async with async_session_factory() as session:
            d = await session.get(Document, doc_uuid)
            a = (
                await session.execute(
                    select(Analysis).where(Analysis.document_id == doc_uuid)
                )
            ).scalar_one_or_none()
            if d:
                d.status = DocumentStatus.error
            if a:
                a.status = AnalysisStatus.error
                a.error_msg = str(exc)[:2000]
                a.completed_at = datetime.now(timezone.utc)
            await session.commit()
        raise

    task.update_state(
        state="PROGRESS",
        meta={"progress": 45, "detail": "Executando pipeline de IA"},
    )

    try:
        pipeline = _get_pipeline()
        result = pipeline.analyze(text, regiao, firm_id=doc.firm_id)
        payload = result.model_dump()
    except Exception as exc:
        logger.exception("Falha no pipeline: %s", exc)
        async with async_session_factory() as session:
            d = await session.get(Document, doc_uuid)
            a = (
                await session.execute(
                    select(Analysis).where(Analysis.document_id == doc_uuid)
                )
            ).scalar_one_or_none()
            if d:
                d.status = DocumentStatus.error
            if a:
                a.status = AnalysisStatus.error
                a.error_msg = str(exc)[:2000]
                a.completed_at = datetime.now(timezone.utc)
            await session.commit()
        raise

    task.update_state(
        state="PROGRESS",
        meta={"progress": 90, "detail": "Persistindo resultado"},
    )

    now = datetime.now(timezone.utc)
    async with async_session_factory() as session:
        d = await session.get(Document, doc_uuid)
        a = (
            await session.execute(
                select(Analysis).where(Analysis.document_id == doc_uuid)
            )
        ).scalar_one_or_none()
        if d:
            d.status = DocumentStatus.done
        if a:
            a.status = AnalysisStatus.done
            a.result = payload
            a.completed_at = now
        await session.commit()


@celery_app.task(bind=True, name="workers.tasks.process_document")
def process_document(self, document_id: str) -> str:
    asyncio.run(_process_document_async(self, document_id))
    return document_id
