from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator
from uuid import UUID

import asyncpg
from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse
from starlette.responses import JSONResponse

from lexai.config import get_settings
from lexai.middleware.security import get_request_user
from lexai.models.chat import ChatHistoryResponse, ChatMessagesResponse, ChatRequest
from lexai.services.chat_history import (
    ensure_session,
    insert_message,
    list_messages_for_session,
    list_sessions,
    touch_session_title_from_first_user_message,
)
from lexai.services.cohere_api import cohere_embed_floats, cohere_rerank_indices
from lexai.services.gemini_service import sse_data, stream_gemini_ui_message_sse
from lexai.services.rag_prompt import (
    build_context_block,
    build_gemini_turn_messages,
    build_system_prompt,
    last_turn_messages,
)
from lexai.services.ui_message import ui_message_to_plain_text
from lexai.services.vector_rag import citations_from_chunks, vector_search_top_k

router = APIRouter(prefix="/api/v1", tags=["chat-v1"])


def get_pool(request: Request) -> asyncpg.Pool:
    return request.app.state.pool


@router.get("/chat/history", response_model=ChatHistoryResponse)
async def chat_history(request: Request, pool: asyncpg.Pool = Depends(get_pool)):
    user = get_request_user(request)
    rows = await list_sessions(pool, user_id=UUID(user["sub"]))
    return {"sessions": rows}


@router.get("/chat/sessions/{session_id}/messages", response_model=ChatMessagesResponse)
async def chat_session_messages(session_id: UUID, request: Request, pool: asyncpg.Pool = Depends(get_pool)):
    user = get_request_user(request)
    rows = await list_messages_for_session(pool, user_id=UUID(user["sub"]), session_id=session_id)
    return {"session_id": session_id, "messages": rows}


@router.post("/chat")
async def chat_v1_rag(body: ChatRequest, request: Request, pool: asyncpg.Pool = Depends(get_pool)):
    settings = get_settings()
    user = get_request_user(request)
    user_id = UUID(user["sub"])
    raw_messages = [m.model_dump(by_alias=True, exclude_none=True) for m in body.messages]

    last_user_text = ""
    for m in reversed(raw_messages):
        if m.get("role") != "user":
            continue
        last_user_text = ui_message_to_plain_text(m)
        break

    if not last_user_text.strip():
        return JSONResponse({"detail": "Última mensagem do usuário está vazia."}, status_code=422)

    try:
        query_embedding = await cohere_embed_floats(
            settings,
            texts=[last_user_text.strip()],
            input_type="search_query",
        )
    except Exception as exc:  # noqa: BLE001
        return JSONResponse({"detail": f"Falha ao gerar embedding: {exc!s}"}, status_code=503)

    async with pool.acquire() as conn:
        top5 = await vector_search_top_k(conn, embedding=query_embedding, k=5)

    if not top5:
        top3 = []
    else:
        try:
            indices = await cohere_rerank_indices(
                settings,
                query=last_user_text.strip(),
                documents=[c.content for c in top5],
                top_n=3,
            )
        except Exception as exc:  # noqa: BLE001
            return JSONResponse({"detail": f"Falha no rerank: {exc!s}"}, status_code=503)
        top3 = [top5[i] for i in indices if 0 <= i < len(top5)]
        if not top3:
            top3 = top5[:3]

    context_chunks = [
        {"id": c.id, "content": c.content, "citation_label": c.citation_label} for c in top3
    ]
    citations_payload = citations_from_chunks(top3)
    context_block = build_context_block(context_chunks)
    system_prompt = build_system_prompt(settings, context_block=context_block)

    turn_slice = last_turn_messages(raw_messages, max_turns=6)
    gemini_contents = build_gemini_turn_messages(turn_slice)
    while gemini_contents and gemini_contents[-1].role == "model":
        gemini_contents.pop()
    if not gemini_contents:
        return JSONResponse({"detail": "Nenhuma mensagem válida para o modelo."}, status_code=422)

    session_id = await ensure_session(
        pool,
        user_id=user_id,
        session_id=body.session_id,
        title_hint=last_user_text[:120],
    )

    await insert_message(pool, session_id=session_id, role="user", content=last_user_text.strip())
    await touch_session_title_from_first_user_message(
        pool,
        session_id=session_id,
        title=last_user_text.strip(),
    )

    assistant_message_id = f"msg_{uuid.uuid4().hex}"
    text_block_id = f"blk_{uuid.uuid4().hex}"

    async def event_stream() -> AsyncIterator[bytes]:
        yield sse_data({"type": "data-lexai-session", "data": {"sessionId": str(session_id)}})
        buffer: list[str] = []
        async for chunk in stream_gemini_ui_message_sse(
            settings,
            contents=gemini_contents,
            assistant_message_id=assistant_message_id,
            text_block_id=text_block_id,
            system=system_prompt,
            citations=citations_payload,
        ):
            yield chunk
            if chunk.startswith(b"data: ") and b"[DONE]" not in chunk:
                line = chunk.decode("utf-8", errors="ignore").removeprefix("data: ").strip()
                if not line or line == "[DONE]":
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if obj.get("type") == "text-delta" and isinstance(obj.get("delta"), str):
                    buffer.append(obj["delta"])

        full = "".join(buffer).strip()
        if full:
            await insert_message(
                pool,
                session_id=session_id,
                role="assistant",
                content=full,
                source_citations=citations_payload,
            )

    headers = {
        "Cache-Control": "no-cache, no-transform",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
        "x-vercel-ai-ui-message-stream": "v1",
        "x-lexai-session-id": str(session_id),
    }
    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream; charset=utf-8",
        headers=headers,
    )
