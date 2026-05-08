"""Documentos."""

from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, Field

from api.models.document import DocumentStatus


class DocumentUploadResponse(BaseModel):
    document_id: uuid.UUID
    filename: str
    status: DocumentStatus


class DocumentResponse(BaseModel):
    id: uuid.UUID
    firm_id: uuid.UUID
    uploaded_by: uuid.UUID | None
    filename: str
    file_path: str
    file_type: str
    size_bytes: int
    status: DocumentStatus
    created_at: datetime

    model_config = {"from_attributes": True}
