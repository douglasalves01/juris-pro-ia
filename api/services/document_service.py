"""Persistência de upload e extração de texto."""

from __future__ import annotations

import re
import uuid
from pathlib import Path

import filetype
from fastapi import HTTPException, UploadFile, status

from api.config import settings
from api.ml.text_extractor import TextExtractor

ALLOWED_MIME = frozenset(
    {
        "application/pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    }
)
_EXT_FOR_MIME = {
    "application/pdf": ".pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
}


def normalize_cnpj(raw: str) -> str:
    digits = re.sub(r"\D", "", raw)
    if len(digits) != 14:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="CNPJ deve conter 14 dígitos",
        )
    return digits


def _sanitize_filename(name: str) -> str:
    base = Path(name).name
    return re.sub(r"[^a-zA-Z0-9._\-]", "_", base)[:200]


async def save_upload(
    firm_id: uuid.UUID,
    upload: UploadFile,
) -> tuple[uuid.UUID, str, str, int, str]:
    """Valida MIME real, tamanho e grava em uploads/{firm_id}/{document_id}.{ext}."""
    doc_id = uuid.uuid4()
    raw_name = upload.filename or "document"
    safe_name = _sanitize_filename(raw_name)
    suffix = Path(safe_name).suffix.lower()
    if suffix not in (".pdf", ".docx"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Extensão permitida: .pdf ou .docx",
        )

    content = await upload.read()
    size = len(content)
    if size > settings.max_upload_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="Arquivo acima do limite de 50MB",
        )
    if size == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Arquivo vazio",
        )

    guessed = filetype.guess(content)
    mime = guessed.mime if guessed else None
    if mime not in ALLOWED_MIME:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Tipo MIME não permitido: {mime or 'desconhecido'}",
        )
    expected_ext = _EXT_FOR_MIME[mime]
    if suffix != expected_ext:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Extensão não confere com o conteúdo do arquivo",
        )

    ext = expected_ext
    base_dir = settings.upload_dir / str(firm_id)
    base_dir.mkdir(parents=True, exist_ok=True)
    rel_path = base_dir / f"{doc_id}{ext}"
    rel_path.write_bytes(content)

    return doc_id, str(rel_path.resolve()), mime, size, ext.lstrip(".")


def extract_text_from_file(file_path: str) -> str:
    extractor = TextExtractor()
    return extractor.extract(file_path)
