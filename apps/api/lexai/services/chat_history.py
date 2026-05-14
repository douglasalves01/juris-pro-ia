from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any
from uuid import UUID

import asyncpg


def _iso(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat()


async def list_messages_for_session(
    pool: asyncpg.Pool,
    *,
    user_id: UUID,
    session_id: UUID,
) -> list[dict[str, Any]]:
    sql = """
        SELECT m.id, m.role, m.content, m.source_citations, m.created_at
        FROM chat_messages m
        INNER JOIN chat_sessions s ON s.id = m.session_id
        WHERE m.session_id = $1::uuid AND s.user_id = $2::uuid
        ORDER BY m.created_at ASC, m.id ASC;
    """
    async with pool.acquire() as conn:
        rows = await conn.fetch(sql, session_id, user_id)
    out: list[dict[str, Any]] = []
    for r in rows:
        role = str(r["role"])
        if role not in ("user", "assistant", "system"):
            role = "user"
        sc = r.get("source_citations", [])
        if isinstance(sc, str):
            try:
                sc = json.loads(sc)
            except json.JSONDecodeError:
                sc = []
        if not isinstance(sc, list):
            sc = []
        out.append(
            {
                "id": int(r["id"]),
                "role": role,
                "content": str(r["content"]),
                "created_at": _iso(r["created_at"]),
                "source_citations": sc,
            }
        )
    return out


async def list_sessions(pool: asyncpg.Pool, *, user_id: UUID, limit: int = 50) -> list[dict[str, Any]]:
    sql = """
        SELECT id::text, title, updated_at
        FROM chat_sessions
        WHERE user_id = $1::uuid
        ORDER BY updated_at DESC
        LIMIT $2::int;
    """
    async with pool.acquire() as conn:
        rows = await conn.fetch(sql, user_id, limit)
    return [
        {"id": r["id"], "title": r["title"] or "Nova conversa", "updated_at": _iso(r["updated_at"])}
        for r in rows
    ]


async def ensure_session(
    pool: asyncpg.Pool,
    *,
    user_id: UUID,
    session_id: UUID | None,
    title_hint: str,
) -> UUID:
    title = (title_hint or "").strip()[:200] or "Nova conversa"
    async with pool.acquire() as conn:
        if session_id is not None:
            row = await conn.fetchrow(
                """
                SELECT id FROM chat_sessions
                WHERE id = $1::uuid AND user_id = $2::uuid;
                """,
                session_id,
                user_id,
            )
            if row is not None:
                await conn.execute(
                    """
                    UPDATE chat_sessions
                    SET updated_at = now()
                    WHERE id = $1::uuid;
                    """,
                    session_id,
                )
                return session_id
        new_id = uuid.uuid4()
        await conn.execute(
            """
            INSERT INTO chat_sessions (id, user_id, title, created_at, updated_at)
            VALUES ($1::uuid, $2::uuid, $3::text, now(), now());
            """,
            new_id,
            user_id,
            title,
        )
        return new_id


async def insert_message(
    pool: asyncpg.Pool,
    *,
    session_id: UUID,
    role: str,
    content: str,
    source_citations: list[dict[str, Any]] | None = None,
) -> None:
    citations_json = json.dumps(source_citations or [], ensure_ascii=False)
    sql = """
        INSERT INTO chat_messages (session_id, role, content, source_citations, created_at)
        VALUES ($1::uuid, $2::text, $3::text, $4::jsonb, now());
    """
    async with pool.acquire() as conn:
        async with conn.transaction():
            await conn.execute(sql, session_id, role, content, citations_json)
            await conn.execute(
                """
                UPDATE chat_sessions SET updated_at = now() WHERE id = $1::uuid;
                """,
                session_id,
            )


async def touch_session_title_from_first_user_message(
    pool: asyncpg.Pool,
    *,
    session_id: UUID,
    title: str,
) -> None:
    t = title.strip()[:200]
    if not t:
        return
    sql = """
        UPDATE chat_sessions
        SET title = CASE
            WHEN title IS NULL OR title = '' OR title = 'Nova conversa' THEN $2::text
            ELSE title
        END,
        updated_at = now()
        WHERE id = $1::uuid;
    """
    async with pool.acquire() as conn:
        await conn.execute(sql, session_id, t)
