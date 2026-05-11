"""Recuperação de casos similares via Qdrant (banco vetorial)."""

from __future__ import annotations

import logging
import os
import re
from typing import Any

import torch
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from sentence_transformers import SentenceTransformer

from api.ml.models._common import get_torch_device, resolve_submodel_path

logger = logging.getLogger(__name__)

COLLECTION = "casos_juridicos"
EMBEDDING_DIM = 768

_CACHE: dict[str, Any] = {}


def _qdrant_client() -> QdrantClient:
    if "qdrant" not in _CACHE:
        host = os.getenv("QDRANT_HOST", "localhost")
        port = int(os.getenv("QDRANT_PORT", "6333"))
        _CACHE["qdrant"] = QdrantClient(host=host, port=port)
    return _CACHE["qdrant"]


def _ensure_model(models_dir: str) -> SentenceTransformer | None:
    path = resolve_submodel_path(models_dir, "embeddings")
    key = str(path)
    if key in _CACHE:
        return _CACHE[key]
    if not path.is_dir():
        logger.warning("Modelo de embeddings não encontrado em: %s", path)
        _CACHE[key] = None
        return None
    device = get_torch_device()
    model = SentenceTransformer(str(path), device=str(device))
    _CACHE[key] = model
    return model


def _collection_exists() -> bool:
    try:
        client = _qdrant_client()
        cols = [c.name for c in client.get_collections().collections]
        if COLLECTION not in cols:
            return False
        info = client.get_collection(COLLECTION)
        return info.points_count > 0
    except Exception as e:
        logger.warning("Qdrant indisponível: %s", e)
        return False


def _terms(text: str) -> set[str]:
    stop = {
        "para", "por", "com", "sem", "uma", "uns", "das", "dos", "que", "não",
        "sao", "são", "foi", "ser", "ter", "art", "artigo", "codigo", "código",
        "processo", "tribunal", "classe", "assuntos",
    }
    return {
        t
        for t in re.findall(r"[a-zA-ZÀ-ÿ0-9]{4,}", (text or "").lower())
        if t not in stop
    }


def _lexical_score(query_terms: set[str], payload: dict[str, Any]) -> float:
    if not query_terms:
        return 0.0
    candidate = " ".join(
        str(payload.get(k) or "")
        for k in ("titulo", "resumo", "tipo", "tribunal", "classe_nome")
    )
    candidate_terms = _terms(candidate)
    if not candidate_terms:
        return 0.0
    overlap = len(query_terms & candidate_terms)
    return min(1.0, overlap / max(8, len(query_terms)))


def predict(
    text: str,
    models_dir: str,
    top_k: int = 5,
    tipo: str | None = None,
    outcome: str | None = None,
    **_kwargs: Any,
) -> dict[str, Any]:
    """
    Busca os top_k casos mais similares no Qdrant.

    Parâmetros opcionais de filtro:
        tipo    — ex. "Consumidor", "Trabalhista"
        outcome — ex. "procedente", "improcedente"
    """
    model = _ensure_model(models_dir)
    if model is None:
        return {
            "similar_cases": [],
            "similar_cases_notice": "Modelo de embeddings não carregado; consulta ao Qdrant não executada.",
        }

    if not _collection_exists():
        return {
            "similar_cases": [],
            "similar_cases_notice": (
                "Coleção Qdrant 'casos_juridicos' indisponível ou vazia. "
                "Importe decisões reais com scripts/fetch_datajud.py, scripts/ingest_casos.py "
                "ou scripts/import_qdrant.py."
            ),
        }

    with torch.inference_mode():
        q_emb = model.encode(
            text or "",
            convert_to_tensor=False,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).tolist()

    # Filtros opcionais por metadados
    conditions = []
    if tipo:
        conditions.append(FieldCondition(key="tipo", match=MatchValue(value=tipo)))
    if outcome:
        conditions.append(FieldCondition(key="outcome", match=MatchValue(value=outcome)))

    query_filter = Filter(must=conditions) if conditions else None

    try:
        client = _qdrant_client()
        # qdrant-client >= 1.10 usa query_points; versões anteriores usam search
        search_limit = max(top_k, min(50, top_k * 4))
        if hasattr(client, "query_points"):
            from qdrant_client.models import Query
            response = client.query_points(
                collection_name=COLLECTION,
                query=q_emb,
                query_filter=query_filter,
                limit=search_limit,
                with_payload=True,
            )
            hits = response.points
        else:
            hits = client.search(
                collection_name=COLLECTION,
                query_vector=q_emb,
                query_filter=query_filter,
                limit=search_limit,
                with_payload=True,
            )
    except Exception as e:
        logger.error("Erro na busca Qdrant: %s", e)
        return {"similar_cases": [], "similar_cases_notice": f"Erro Qdrant: {e}"}

    query_terms = _terms(text)
    ranked = []
    for hit in hits:
        payload = hit.payload or {}
        vector_score = float(hit.score)
        lexical = _lexical_score(query_terms, payload)
        rerank_score = (0.85 * vector_score) + (0.15 * lexical)
        ranked.append((rerank_score, vector_score, lexical, hit, payload))
    ranked.sort(key=lambda item: item[0], reverse=True)

    public_results = [
        {
            "id": str(hit.id),
            "number": str(payload.get("numeroProcesso", "")),
            "numeroProcesso": str(payload.get("numeroProcesso", "")),
            "titulo": str(payload.get("titulo", "")),
            "resumo": str(payload.get("resumo", "")),
            "outcome": str(payload.get("outcome", "")),
            "tipo": str(payload.get("tipo", "")),
            "tribunal": str(payload.get("tribunal", "")),
            "similaridade": round(rerank_score, 4),
            "vector_score": round(vector_score, 4),
            "lexical_score": round(lexical, 4),
        }
        for rerank_score, vector_score, lexical, hit, payload in ranked[:top_k]
    ]

    return {"similar_cases": public_results[:top_k], "similar_cases_notice": None}
