from __future__ import annotations

import asyncpg

from lexai.config import Settings

_SCHEMA_STATEMENTS: tuple[str, ...] = (
    """
    CREATE EXTENSION IF NOT EXISTS pgcrypto;
    """,
    """
    CREATE EXTENSION IF NOT EXISTS vector;
    """,
    """
    CREATE TABLE IF NOT EXISTS chat_sessions (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        user_id UUID NOT NULL,
        title TEXT NOT NULL DEFAULT '',
        created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
        updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS chat_messages (
        id BIGSERIAL PRIMARY KEY,
        session_id UUID NOT NULL REFERENCES chat_sessions(id) ON DELETE CASCADE,
        role TEXT NOT NULL,
        content TEXT NOT NULL,
        source_citations JSONB NOT NULL DEFAULT '[]'::jsonb,
        created_at TIMESTAMPTZ NOT NULL DEFAULT now()
    );
    """,
    """
    ALTER TABLE chat_messages
        ADD COLUMN IF NOT EXISTS source_citations JSONB NOT NULL DEFAULT '[]'::jsonb;
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_chat_messages_session_created
        ON chat_messages (session_id, created_at);
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_chat_sessions_user_updated
        ON chat_sessions (user_id, updated_at DESC);
    """,
    """
    CREATE TABLE IF NOT EXISTS legal_rag_chunks (
        id BIGSERIAL PRIMARY KEY,
        content TEXT NOT NULL,
        citation_label TEXT NOT NULL,
        embedding vector(1024) NOT NULL
    );
    """,
    """
    CREATE INDEX IF NOT EXISTS legal_rag_chunks_embedding_hnsw
        ON legal_rag_chunks
        USING hnsw (embedding vector_cosine_ops);
    """,
)


async def create_pool(settings: Settings) -> asyncpg.Pool:
    pool = await asyncpg.create_pool(settings.database_url, min_size=1, max_size=10)
    async with pool.acquire() as conn:
        for stmt in _SCHEMA_STATEMENTS:
            try:
                await conn.execute(stmt)
            except asyncpg.UndefinedObjectError:
                if "vector" in stmt.lower():
                    raise RuntimeError(
                        "Extensão pgvector indisponível. Use imagem Postgres com pgvector "
                        "(ex.: pgvector/pgvector:pg16) ou instale a extensão no servidor."
                    ) from None
                raise
    return pool
