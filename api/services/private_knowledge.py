"""RAG privado por escritorio com Qdrant opcional e fallback local."""

from __future__ import annotations

import math
import os
import socket
import time
import uuid
from collections import Counter
from dataclasses import dataclass
from hashlib import sha256
from typing import Any


_EMBEDDING_DIM = 384
_QDRANT_CLIENT: Any | None = None
_QDRANT_DISABLED_UNTIL = 0.0
_READY_COLLECTIONS: set[str] = set()


@dataclass(frozen=True)
class KnowledgeEntry:
    id: str
    firm_id: str
    title: str
    text: str
    type: str
    source_url: str | None
    vector: list[float]
    inserted_at: float


_LOCAL: dict[str, list[KnowledgeEntry]] = {}
_UPDATED_AT: dict[str, float] = {}


def collection_name(firm_id: str | uuid.UUID) -> str:
    fid = str(uuid.UUID(str(firm_id)))
    return f"firm_{fid}"


def _tokens(text: str) -> list[str]:
    import re

    return [token.lower() for token in re.findall(r"[a-zA-ZÀ-ÿ0-9]{4,}", text or "")]


def embed(text: str) -> list[float]:
    counts = Counter(_tokens(text))
    vector = [0.0] * _EMBEDDING_DIM
    for token, count in counts.items():
        index = int.from_bytes(sha256(token.encode("utf-8")).digest()[:4], "big") % _EMBEDDING_DIM
        vector[index] += float(count)
    norm = math.sqrt(sum(value * value for value in vector)) or 1.0
    return [value / norm for value in vector]


def cosine(left: list[float], right: list[float]) -> float:
    if not left or not right:
        return 0.0
    return sum(l * r for l, r in zip(left, right, strict=False))


def _qdrant_client() -> Any | None:
    global _QDRANT_CLIENT, _QDRANT_DISABLED_UNTIL
    if os.getenv("JURISPRO_PRIVATE_RAG_BACKEND", "auto").strip().lower() == "memory":
        return None
    if time.time() < _QDRANT_DISABLED_UNTIL:
        return None
    if _QDRANT_CLIENT is None:
        try:
            from qdrant_client import QdrantClient

            host = os.getenv("QDRANT_HOST", "localhost")
            port = int(os.getenv("QDRANT_PORT", "6333"))
            with socket.create_connection((host, port), timeout=0.25):
                pass
            _QDRANT_CLIENT = QdrantClient(host=host, port=port, check_compatibility=False)
        except Exception:
            _QDRANT_DISABLED_UNTIL = time.time() + 60
            return None
    return _QDRANT_CLIENT


def _ensure_collection(client: Any, collection: str) -> bool:
    global _QDRANT_DISABLED_UNTIL
    if collection in _READY_COLLECTIONS:
        return True
    try:
        from qdrant_client.models import Distance, VectorParams

        collections = [item.name for item in client.get_collections().collections]
        if collection not in collections:
            client.create_collection(
                collection_name=collection,
                vectors_config=VectorParams(size=_EMBEDDING_DIM, distance=Distance.COSINE),
            )
        _READY_COLLECTIONS.add(collection)
        return True
    except Exception:
        _QDRANT_DISABLED_UNTIL = time.time() + 60
        return False


def ingest_documents(firm_id: str, documents: list[dict[str, Any]]) -> int:
    collection = collection_name(firm_id)
    now = time.time()
    entries: list[KnowledgeEntry] = []
    points = []
    for idx, doc in enumerate(documents):
        text = str(doc.get("text") or "").strip()
        if not text:
            continue
        doc_id = str(doc.get("documentId") or uuid.uuid4())
        try:
            point_id = str(uuid.UUID(doc_id))
        except ValueError:
            point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{firm_id}:{doc_id}"))
        title = str(doc.get("title") or f"Documento {idx + 1}")
        vector = embed(title + "\n\n" + text)
        entry = KnowledgeEntry(
            id=doc_id,
            firm_id=str(uuid.UUID(str(firm_id))),
            title=title,
            text=text,
            type=str(doc.get("type") or "outro"),
            source_url=doc.get("sourceUrl"),
            vector=vector,
            inserted_at=now,
        )
        entries.append(entry)
        points.append(
            {
                "id": point_id,
                "vector": vector,
                "payload": {
                    "documentId": doc_id,
                    "firmId": entry.firm_id,
                    "title": entry.title,
                    "text": entry.text[:8000],
                    "type": entry.type,
                    "sourceUrl": entry.source_url,
                    "insertedAt": now,
                },
            }
        )

    if not entries:
        return 0

    _LOCAL.setdefault(collection, []).extend(entries)
    _UPDATED_AT[collection] = now

    client = _qdrant_client()
    if client is None or not _ensure_collection(client, collection):
        return len(entries)
    try:
        from qdrant_client.models import PointStruct

        client.upsert(
            collection_name=collection,
            points=[
                PointStruct(id=item["id"], vector=item["vector"], payload=item["payload"])
                for item in points
            ],
        )
    except Exception:
        pass
    return len(entries)


def search_private(firm_id: str | uuid.UUID | None, text: str, top_k: int = 3) -> list[dict[str, Any]]:
    if not firm_id:
        return []
    collection = collection_name(firm_id)
    vector = embed(text)
    hits = _qdrant_search(collection, vector, top_k)
    if hits:
        return hits
    ranked = []
    for entry in _LOCAL.get(collection, []):
        ranked.append((cosine(vector, entry.vector), entry))
    ranked.sort(key=lambda item: item[0], reverse=True)
    return [
        {
            "id": entry.id,
            "number": "",
            "numeroProcesso": "",
            "titulo": entry.title,
            "resumo": entry.text[:500],
            "outcome": "",
            "tipo": entry.type,
            "tribunal": "Acervo privado",
            "similaridade": round(score, 4),
            "source": "private_rag",
        }
        for score, entry in ranked[:top_k]
        if score > 0.0
    ]


def _qdrant_search(collection: str, vector: list[float], top_k: int) -> list[dict[str, Any]]:
    client = _qdrant_client()
    if client is None or not _ensure_collection(client, collection):
        return []
    try:
        if hasattr(client, "query_points"):
            response = client.query_points(
                collection_name=collection,
                query=vector,
                limit=top_k,
                with_payload=True,
            )
            hits = response.points
        else:
            hits = client.search(
                collection_name=collection,
                query_vector=vector,
                limit=top_k,
                with_payload=True,
            )
    except Exception:
        return []
    results = []
    for hit in hits:
        payload = hit.payload or {}
        results.append(
            {
                "id": str(payload.get("documentId") or hit.id),
                "number": "",
                "numeroProcesso": "",
                "titulo": str(payload.get("title") or ""),
                "resumo": str(payload.get("text") or "")[:500],
                "outcome": "",
                "tipo": str(payload.get("type") or ""),
                "tribunal": "Acervo privado",
                "similaridade": round(float(hit.score), 4),
                "source": "private_rag",
            }
        )
    return results


def few_shot_context(firm_id: str | None, query: str, top_k: int = 3) -> list[dict[str, str]]:
    return [
        {"title": item["titulo"], "text": item["resumo"]}
        for item in search_private(firm_id, query, top_k=top_k)
    ]


def stats(firm_id: str) -> dict[str, Any]:
    collection = collection_name(firm_id)
    count = len(_LOCAL.get(collection, []))
    updated = _UPDATED_AT.get(collection)
    client = _qdrant_client()
    if client is not None and _ensure_collection(client, collection):
        try:
            count = int(client.get_collection(collection).points_count or count)
        except Exception:
            pass
    return {
        "firmId": str(uuid.UUID(str(firm_id))),
        "collection": collection,
        "documentsIndexed": count,
        "lastUpdatedAt": (
            time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(updated))
            if updated
            else None
        ),
    }


def clear() -> None:
    _LOCAL.clear()
    _UPDATED_AT.clear()
