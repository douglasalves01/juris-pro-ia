from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import asyncpg


def format_vector_literal(values: list[float]) -> str:
    return "[" + ",".join(f"{v:.8f}" for v in values) + "]"


@dataclass(frozen=True)
class RagChunk:
    id: int
    content: str
    citation_label: str


async def vector_search_top_k(
    conn: asyncpg.Connection,
    *,
    embedding: list[float],
    k: int = 5,
) -> list[RagChunk]:
    vec = format_vector_literal(embedding)
    sql = """
        SELECT id, content, citation_label
        FROM legal_rag_chunks
        ORDER BY embedding <=> $1::vector
        LIMIT $2::int;
    """
    rows = await conn.fetch(sql, vec, k)
    return [
        RagChunk(id=int(r["id"]), content=str(r["content"]), citation_label=str(r["citation_label"]))
        for r in rows
    ]


def citations_from_chunks(chunks: list[RagChunk]) -> list[dict[str, Any]]:
    return [
        {
            "label": c.citation_label,
            "chunk_id": c.id,
            "snippet": (c.content[:280] + "…") if len(c.content) > 280 else c.content,
        }
        for c in chunks
    ]
