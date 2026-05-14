from __future__ import annotations

from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, Field, Field


class TextPart(BaseModel):
    type: Literal["text"] = "text"
    text: str


class UIMessage(BaseModel):
    id: str | None = None
    role: Literal["user", "assistant", "system"]
    parts: list[dict[str, Any]] | None = None
    content: str | list[dict[str, Any]] | None = None


class ChatSessionSummary(BaseModel):
    id: UUID
    title: str
    updated_at: str


class ChatMessageRow(BaseModel):
    id: int
    role: Literal["user", "assistant", "system"]
    content: str
    created_at: str
    source_citations: list[dict[str, Any]] = Field(default_factory=list)


class ChatMessagesResponse(BaseModel):
    session_id: UUID
    messages: list[ChatMessageRow]


class ChatHistoryResponse(BaseModel):
    sessions: list[ChatSessionSummary]


class ChatRequest(BaseModel):
    messages: list[UIMessage]
    session_id: UUID | None = Field(default=None, alias="sessionId")

    model_config = {"populate_by_name": True}
