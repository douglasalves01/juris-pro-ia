from __future__ import annotations

import json
from collections.abc import AsyncIterator

from google import genai
from google.genai import types

from lexai.config import Settings
from lexai.services.ui_message import ui_message_to_plain_text


def build_gemini_turn_contents(messages: list[dict]) -> list[types.Content]:
    out: list[types.Content] = []
    for m in messages:
        role = m.get("role")
        if role == "user":
            gemini_role = "user"
        elif role == "assistant":
            gemini_role = "model"
        else:
            continue
        text = ui_message_to_plain_text(m).strip()
        if not text:
            continue
        out.append(types.Content(role=gemini_role, parts=[types.Part.from_text(text=text)]))
    return out


def sse_data(obj: dict | str) -> bytes:
    if isinstance(obj, str):
        return f"data: {obj}\n\n".encode("utf-8")
    return ("data: " + json.dumps(obj, ensure_ascii=False) + "\n\n").encode("utf-8")


def _stream_text_delta(chunk: types.GenerateContentResponse, *, seen_full: str) -> tuple[str, str]:
    """Returns (delta, new_seen_full) from a streaming chunk."""
    full = chunk.text or ""
    if not full:
        return "", seen_full
    if seen_full and full.startswith(seen_full):
        return full[len(seen_full) :], full
    if not seen_full:
        return full, full
    # Chunk may carry only a new fragment
    return full, seen_full + full


async def stream_gemini_ui_message_sse(
    settings: Settings,
    *,
    contents: list[types.Content],
    assistant_message_id: str,
    text_block_id: str,
    system: str | None = None,
    citations: list[dict] | None = None,
) -> AsyncIterator[bytes]:
    if not settings.gemini_api_key.strip():
        yield sse_data({"type": "error", "errorText": "GEMINI_API_KEY is not configured."})
        yield sse_data("[DONE]")
        return

    if not contents:
        yield sse_data({"type": "error", "errorText": "No messages to send to Gemini."})
        yield sse_data("[DONE]")
        return

    client = genai.Client(api_key=settings.gemini_api_key)
    yield sse_data({"type": "start", "messageId": assistant_message_id})
    yield sse_data({"type": "text-start", "id": text_block_id})

    cfg_kwargs: dict = {"max_output_tokens": 8192}
    if system and system.strip():
        cfg_kwargs["system_instruction"] = system
    config = types.GenerateContentConfig(**cfg_kwargs)

    seen = ""
    try:
        stream = await client.aio.models.generate_content_stream(
            model=settings.gemini_model,
            contents=contents,
            config=config,
        )
        async for chunk in stream:
            delta, seen = _stream_text_delta(chunk, seen_full=seen)
            if delta:
                yield sse_data({"type": "text-delta", "id": text_block_id, "delta": delta})
    except Exception as exc:  # noqa: BLE001
        yield sse_data({"type": "error", "errorText": str(exc)})
        yield sse_data({"type": "text-end", "id": text_block_id})
        yield sse_data({"type": "finish"})
        yield sse_data("[DONE]")
        return

    yield sse_data({"type": "text-end", "id": text_block_id})
    if citations:
        yield sse_data({"type": "data-lexai-citations", "data": {"citations": citations}})
    yield sse_data({"type": "finish"})
    yield sse_data("[DONE]")
