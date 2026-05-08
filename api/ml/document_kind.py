"""Detecção heurística do gênero documental (contrato x peça processual)."""

from __future__ import annotations

import re
from typing import Literal

DocumentKind = Literal["peticao_inicial", "contrato", "outro"]

_PETICAO_HINTS = [
    r"\bvem\s+prop",
    r"\brequerente\b",
    r"\brequerid[oa]\b",
    r"\bDOS\s+FATOS\b",
    r"\bDOS\s+PEDIDOS\b",
    r"\bEXCELENT[ÍI]SS",
    r"\bVARA\s+",
    r"\btutela\s+antecipad",
    r"\bpedido\s+de\s+cita",
    r"\btermos\s+em\s+que",
    r"\bDIREITO\b.*\bJUIZ\b",
]
_CONTRATO_HINTS = [
    r"\bCL[ÁA]USULA\s+(?:PRIMEIRA|\d+|[IVXLCDM]+)",
    r"\bcontratante\b",
    r"\bcontratad[oa]\b",
    r"\bCONTRATO\s+DE\s+",
    r"\bDO\s+OBJETO\b",
    r"\bELEI[ÇC][ÃA]O\s+DE\s+FORO\b",
]


def detect_document_kind(text: str) -> DocumentKind:
    """Classifica o texto para adaptar regras de risco, sumário e classificação."""
    t = (text or "").strip()
    if len(t) < 120:
        return "outro"
    pet = sum(1 for p in _PETICAO_HINTS if re.search(p, t, re.I))
    con = sum(1 for p in _CONTRATO_HINTS if re.search(p, t, re.I))
    if pet >= 2 and pet >= con:
        return "peticao_inicial"
    if con >= 2 and con > pet:
        return "contrato"
    if pet >= 1 and con == 0:
        return "peticao_inicial"
    if con >= 1 and pet == 0:
        return "contrato"
    if pet > con:
        return "peticao_inicial"
    if con > pet:
        return "contrato"
    return "outro"
