"""Análise assíncrona e consulta de resultados."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from api.database import get_db
from api.dependencies import require_roles
from api.models.document import (
    Analysis,
    AnalysisStatus,
    Document,
    DocumentStatus,
)
from api.models.user import User, UserRole
from api.schemas.analysis import (
    AnalysisResponse,
    AnalysisStartResponse,
    AnalysisStatusResponse,
)
from api.services.analysis_service import enqueue_process_document, get_task_progress

router = APIRouter(prefix="/analysis", tags=["analysis"])


@router.post("/start/{doc_id}", response_model=AnalysisStartResponse)
async def start_analysis(
    doc_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current: User = Depends(require_roles(UserRole.admin, UserRole.advogado)),
) -> AnalysisStartResponse:
    doc = await db.get(Document, doc_id)
    if doc is None or doc.firm_id != current.firm_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Documento não encontrado",
        )

    r = await db.execute(
        select(Analysis).where(Analysis.document_id == doc_id)
    )
    analysis = r.scalar_one_or_none()
    if analysis is None:
        analysis = Analysis(
            id=uuid.uuid4(),
            document_id=doc_id,
            status=AnalysisStatus.pending,
        )
        db.add(analysis)
        await db.flush()
    else:
        analysis.status = AnalysisStatus.pending
        analysis.result = None
        analysis.error_msg = None
        analysis.completed_at = None

    doc.status = DocumentStatus.processing

    task_id = enqueue_process_document(doc_id)
    analysis.task_id = task_id
    await db.flush()

    return AnalysisStartResponse(task_id=task_id, analysis_id=analysis.id)


@router.get("/status/{task_id}", response_model=AnalysisStatusResponse)
async def analysis_task_status(
    task_id: str,
    _: User = Depends(
        require_roles(UserRole.admin, UserRole.advogado, UserRole.secretaria)
    ),
) -> AnalysisStatusResponse:
    st, progress, detail = get_task_progress(task_id)
    return AnalysisStatusResponse(status=st, progress=progress, detail=detail)


@router.get("/{analysis_id}", response_model=AnalysisResponse)
async def get_analysis(
    analysis_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current: User = Depends(
        require_roles(UserRole.admin, UserRole.advogado, UserRole.secretaria)
    ),
) -> Analysis:
    r = await db.execute(
        select(Analysis)
        .options(selectinload(Analysis.document))
        .where(Analysis.id == analysis_id)
    )
    analysis = r.scalar_one_or_none()
    if analysis is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Análise não encontrada",
        )
    doc = analysis.document
    if doc is None or doc.firm_id != current.firm_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Análise não encontrada",
        )
    return analysis
