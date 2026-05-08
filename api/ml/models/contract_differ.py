"""Comparação semântica de versões de contrato."""

from __future__ import annotations

import time
from typing import Any

import torch
from pydantic import BaseModel

from api.ml.models import case_retriever
from api.ml.preprocessor import TextPreprocessor


class ClauseDiff(BaseModel):
    numero_a: str
    numero_b: str | None
    tipo: str
    status: str
    similaridade: float
    texto_a: str
    texto_b: str | None


class CompareResult(BaseModel):
    total_clausulas_a: int
    total_clausulas_b: int
    mantidas: int
    modificadas: int
    removidas: int
    adicionadas: int
    diffs: list[ClauseDiff]
    processing_time_seconds: float


def compare_texts(
    text_a: str,
    text_b: str,
    preprocessor: TextPreprocessor,
    models_dir: str,
) -> CompareResult:
    t0 = time.perf_counter()
    clauses_a = preprocessor.extract_clauses(text_a or "")
    clauses_b = preprocessor.extract_clauses(text_b or "")

    texts_a = [_clause_text(c) for c in clauses_a]
    texts_b = [_clause_text(c) for c in clauses_b]
    embeddings_a = _encode(texts_a, models_dir)
    embeddings_b = _encode(texts_b, models_dir)

    used_b: set[int] = set()
    diffs: list[ClauseDiff] = []

    for idx_a, clause_a in enumerate(clauses_a):
        best_idx: int | None = None
        best_score = -1.0
        for idx_b in range(len(clauses_b)):
            if idx_b in used_b:
                continue
            score = _cosine(embeddings_a[idx_a], embeddings_b[idx_b])
            if score > best_score:
                best_score = score
                best_idx = idx_b

        if best_idx is None or best_score < 0.50:
            diffs.append(_diff(clause_a, None, "removida", max(best_score, 0.0)))
            continue

        used_b.add(best_idx)
        status = "mantida" if best_score > 0.90 else "modificada"
        diffs.append(_diff(clause_a, clauses_b[best_idx], status, best_score))

    for idx_b, clause_b in enumerate(clauses_b):
        if idx_b in used_b:
            continue
        diffs.append(
            ClauseDiff(
                numero_a="",
                numero_b=str(clause_b.get("numero", "")),
                tipo=str(clause_b.get("tipo", "geral")),
                status="adicionada",
                similaridade=0.0,
                texto_a="",
                texto_b=str(clause_b.get("texto", "")),
            )
        )

    counts = {
        status: sum(1 for d in diffs if d.status == status)
        for status in ("mantida", "modificada", "removida", "adicionada")
    }
    return CompareResult(
        total_clausulas_a=len(clauses_a),
        total_clausulas_b=len(clauses_b),
        mantidas=counts["mantida"],
        modificadas=counts["modificada"],
        removidas=counts["removida"],
        adicionadas=counts["adicionada"],
        diffs=diffs,
        processing_time_seconds=round(time.perf_counter() - t0, 4),
    )


def _encode(texts: list[str], models_dir: str) -> list[list[float]]:
    if not texts:
        return []
    model = case_retriever._ensure_model(models_dir)
    if model is None:
        raise RuntimeError("Modelo de embeddings não carregado.")
    with torch.inference_mode():
        encoded = model.encode(
            texts,
            convert_to_tensor=False,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
    return [list(map(float, row)) for row in encoded]


def _clause_text(clause: dict[str, Any]) -> str:
    title = str(clause.get("titulo", "")).strip()
    body = str(clause.get("texto", "")).strip()
    return f"{title}\n{body}".strip()


def _cosine(vec_a: list[float], vec_b: list[float]) -> float:
    if not vec_a or not vec_b:
        return 0.0
    size = min(len(vec_a), len(vec_b))
    return round(sum(vec_a[i] * vec_b[i] for i in range(size)), 4)


def _diff(
    clause_a: dict[str, Any],
    clause_b: dict[str, Any] | None,
    status: str,
    similarity: float,
) -> ClauseDiff:
    return ClauseDiff(
        numero_a=str(clause_a.get("numero", "")),
        numero_b=str(clause_b.get("numero", "")) if clause_b else None,
        tipo=str(
            clause_a.get(
                "tipo",
                clause_b.get("tipo", "geral") if clause_b else "geral",
            )
        ),
        status=status,
        similaridade=round(similarity, 4),
        texto_a=str(clause_a.get("texto", "")),
        texto_b=str(clause_b.get("texto", "")) if clause_b else None,
    )
