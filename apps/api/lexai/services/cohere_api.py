from __future__ import annotations

from typing import Any

import httpx

from lexai.config import Settings


async def cohere_embed_floats(settings: Settings, *, texts: list[str], input_type: str) -> list[list[float]]:
    if not settings.cohere_api_key.strip():
        raise RuntimeError("COHERE_API_KEY is not configured.")
    url = "https://api.cohere.ai/v1/embed"
    payload: dict[str, Any] = {
        "texts": texts,
        "model": settings.cohere_embed_model,
        "input_type": input_type,
        "embedding_types": ["float"],
    }
    headers = {
        "Authorization": f"Bearer {settings.cohere_api_key}",
        "Content-Type": "application/json",
    }
    async with httpx.AsyncClient(timeout=60.0) as client:
        r = await client.post(url, json=payload, headers=headers)
        r.raise_for_status()
        data = r.json()
    emb = data.get("embeddings", {}).get("float")
    if not isinstance(emb, list) or not emb:
        raise RuntimeError("Cohere embed response missing embeddings.float")
    row0 = emb[0]
    if not isinstance(row0, list):
        raise RuntimeError("Invalid embedding row")
    return [float(x) for x in row0]


async def cohere_rerank_indices(
    settings: Settings,
    *,
    query: str,
    documents: list[str],
    top_n: int,
) -> list[int]:
    if not settings.cohere_api_key.strip():
        raise RuntimeError("COHERE_API_KEY is not configured.")
    url = "https://api.cohere.ai/v1/rerank"
    payload = {
        "model": settings.cohere_rerank_model,
        "query": query,
        "documents": documents,
        "top_n": min(top_n, len(documents)),
    }
    headers = {
        "Authorization": f"Bearer {settings.cohere_api_key}",
        "Content-Type": "application/json",
    }
    async with httpx.AsyncClient(timeout=60.0) as client:
        r = await client.post(url, json=payload, headers=headers)
        r.raise_for_status()
        data = r.json()
    results = data.get("results")
    if not isinstance(results, list):
        return []
    out: list[int] = []
    for item in results:
        if isinstance(item, dict) and isinstance(item.get("index"), int):
            out.append(int(item["index"]))
    return out
