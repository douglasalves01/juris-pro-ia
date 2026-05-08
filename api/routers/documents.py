"""Upload e listagem de documentos."""

from __future__ import annotations

import os
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api.database import get_db
from api.dependencies import require_roles
from api.models.document import Document, DocumentStatus
from api.models.user import User, UserRole
from api.schemas.document import DocumentResponse, DocumentUploadResponse
from api.services.document_service import save_upload

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
    current: User = Depends(require_roles(UserRole.admin, UserRole.advogado)),
) -> DocumentUploadResponse:
    doc_id, file_path, mime, size, ftype = await save_upload(current.firm_id, file)
    doc = Document(
        id=doc_id,
        firm_id=current.firm_id,
        uploaded_by=current.id,
        filename=file.filename or f"{doc_id}.{ftype}",
        file_path=file_path,
        file_type=mime,
        size_bytes=size,
        status=DocumentStatus.pending,
    )
    db.add(doc)
    await db.flush()
    return DocumentUploadResponse(
        document_id=doc.id,
        filename=doc.filename,
        status=doc.status,
    )


@router.get("/", response_model=list[DocumentResponse])
async def list_documents(
    db: AsyncSession = Depends(get_db),
    current: User = Depends(
        require_roles(UserRole.admin, UserRole.advogado, UserRole.secretaria)
    ),
) -> list[Document]:
    r = await db.execute(
        select(Document).where(Document.firm_id == current.firm_id)
    )
    return list(r.scalars().all())


@router.get("/{document_id}", response_model=DocumentResponse)
async def get_document(
    document_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current: User = Depends(
        require_roles(UserRole.admin, UserRole.advogado, UserRole.secretaria)
    ),
) -> Document:
    doc = await db.get(Document, document_id)
    if doc is None or doc.firm_id != current.firm_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Documento não encontrado",
        )
    return doc


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(
    document_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current: User = Depends(require_roles(UserRole.admin, UserRole.advogado)),
) -> None:
    doc = await db.get(Document, document_id)
    if doc is None or doc.firm_id != current.firm_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Documento não encontrado",
        )
    path = Path(doc.file_path)
    if path.is_file():
        try:
            os.remove(path)
        except OSError:
            pass
    db.delete(doc)
    await db.flush()
