"""Cache semantico leve para evitar chamadas repetidas a LLM."""

from __future__ import annotations

import math
import os
import socket
import uuid
import time
from collections import Counter
from dataclasses import dataclass
from hashlib import sha256
from typing import Any


@dataclass(frozen=True)
class SemanticCacheEntry:
    key: str
    mode: str
    contract_type: str
    text: str
    vector: list[float]
    inserted_at: float


_CACHE: list[SemanticCacheEntry] = []
_TTL_SECONDS = 7 * 24 * 60 * 60
_MAX_ENTRIES = 200
_STATS = {"hits": 0, "misses": 0}
_COLLECTION = "semantic_cache"
_EMBEDDING_DIM = 384
_QDRANT_DISABLED_UNTIL = 0.0
_QDRANT_CLIENT: Any | None = None
_QDRANT_COLLECTION_READY = False


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


def threshold() -> float:
    return float(os.getenv("JURISPRO_SEMANTIC_CACHE_THRESHOLD", "0.96"))


def _qdrant_enabled() -> bool:
    backend = os.getenv("JURISPRO_SEMANTIC_CACHE_BACKEND", "auto").strip().lower()
    return backend != "memory" and time.time() >= _QDRANT_DISABLED_UNTIL


def _disable_qdrant_temporarily() -> None:
    global _QDRANT_DISABLED_UNTIL
    _QDRANT_DISABLED_UNTIL = time.time() + 60


def _qdrant_client() -> Any | None:
    global _QDRANT_CLIENT
    if not _qdrant_enabled():
        return None
    if _QDRANT_CLIENT is None:
        try:
            from qdrant_client import QdrantClient

            host = os.getenv("QDRANT_HOST", "localhost")
            port = int(os.getenv("QDRANT_PORT", "6333"))
            with socket.create_connection((host, port), timeout=0.25):
                pass
            _QDRANT_CLIENT = QdrantClient(
                host=host,
                port=port,
                check_compatibility=False,
            )
        except Exception:
            _disable_qdrant_temporarily()
            return None
    return _QDRANT_CLIENT


def _ensure_qdrant_collection(client: Any) -> bool:
    global _QDRANT_COLLECTION_READY
    if _QDRANT_COLLECTION_READY:
        return True
    try:
        from qdrant_client.models import Distance, VectorParams

        collections = [item.name for item in client.get_collections().collections]
        if _COLLECTION not in collections:
            client.create_collection(
                collection_name=_COLLECTION,
                vectors_config=VectorParams(size=_EMBEDDING_DIM, distance=Distance.COSINE),
            )
        _QDRANT_COLLECTION_READY = True
        return True
    except Exception:
        _disable_qdrant_temporarily()
        return False


def _memory_get(mode: str, contract_type: str, vector: list[float], now: float) -> str | None:
    best: tuple[float, SemanticCacheEntry] | None = None
    _CACHE[:] = [entry for entry in _CACHE if now - entry.inserted_at <= _TTL_SECONDS]
    for entry in _CACHE:
        if entry.mode != mode or entry.contract_type != contract_type:
            continue
        score = cosine(vector, entry.vector)
        if best is None or score > best[0]:
            best = (score, entry)
    if best is not None and best[0] >= threshold():
        return best[1].text
    return None


def _qdrant_get(mode: str, contract_type: str, vector: list[float], now: float) -> str | None:
    client = _qdrant_client()
    if client is None or not _ensure_qdrant_collection(client):
        return None
    try:
        from qdrant_client.models import FieldCondition, Filter, MatchValue, Range

        query_filter = Filter(
            must=[
                FieldCondition(key="mode", match=MatchValue(value=mode)),
                FieldCondition(key="contractType", match=MatchValue(value=contract_type)),
                FieldCondition(key="expiresAt", range=Range(gte=now)),
            ]
        )
        if hasattr(client, "query_points"):
            response = client.query_points(
                collection_name=_COLLECTION,
                query=vector,
                query_filter=query_filter,
                limit=1,
                with_payload=True,
            )
            hits = response.points
        else:
            hits = client.search(
                collection_name=_COLLECTION,
                query_vector=vector,
                query_filter=query_filter,
                limit=1,
                with_payload=True,
            )
    except Exception:
        _disable_qdrant_temporarily()
        return None

    if not hits:
        return None
    hit = hits[0]
    if float(hit.score) < threshold():
        return None
    payload = hit.payload or {}
    cached_text = payload.get("responseText")
    return str(cached_text) if cached_text else None


def get(mode: str, contract_type: str, text: str) -> str | None:
    now = time.time()
    vector = embed(text)
    hit = _qdrant_get(mode, contract_type, vector, now) or _memory_get(
        mode,
        contract_type,
        vector,
        now,
    )
    if hit is not None:
        _STATS["hits"] += 1
        return hit
    _STATS["misses"] += 1
    return None


def put(key: str, mode: str, contract_type: str, source_text: str, response_text: str) -> None:
    vector = embed(source_text)
    now = time.time()
    _CACHE.append(
        SemanticCacheEntry(
            key=key,
            mode=mode,
            contract_type=contract_type,
            text=response_text,
            vector=vector,
            inserted_at=now,
        )
    )
    if len(_CACHE) > _MAX_ENTRIES:
        del _CACHE[:-_MAX_ENTRIES]
    _qdrant_put(key, mode, contract_type, source_text, response_text, vector, now)


def _qdrant_put(
    key: str,
    mode: str,
    contract_type: str,
    source_text: str,
    response_text: str,
    vector: list[float],
    now: float,
) -> None:
    client = _qdrant_client()
    if client is None or not _ensure_qdrant_collection(client):
        return
    try:
        from qdrant_client.models import PointStruct

        client.upsert(
            collection_name=_COLLECTION,
            points=[
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vector,
                    payload={
                        "key": key,
                        "mode": mode,
                        "contractType": contract_type,
                        "sourceText": source_text[:4000],
                        "responseText": response_text,
                        "insertedAt": now,
                        "expiresAt": now + _TTL_SECONDS,
                    },
                )
            ],
        )
    except Exception:
        _disable_qdrant_temporarily()


def stats() -> dict[str, float | int]:
    total = _STATS["hits"] + _STATS["misses"]
    return {
        "cacheHits": _STATS["hits"],
        "cacheMisses": _STATS["misses"],
        "cacheHitRate": round(_STATS["hits"] / total, 4) if total else 0.0,
        "cacheEntries": len(_CACHE),
    }


def clear() -> None:
    _CACHE.clear()
    _STATS["hits"] = 0
    _STATS["misses"] = 0
