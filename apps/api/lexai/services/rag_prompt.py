from __future__ import annotations

from typing import Any

from lexai.config import Settings
from lexai.services.gemini_service import build_gemini_turn_contents


def last_turn_messages(messages: list[dict], *, max_turns: int = 6) -> list[dict]:
    core = [m for m in messages if m.get("role") in ("user", "assistant")]
    cap = max_turns * 2
    return core[-cap:] if len(core) > cap else core


def build_context_block(chunks: list[dict[str, Any]]) -> str:
    if not chunks:
        return "(Nenhum trecho recuperado do acervo interno.)"
    parts: list[str] = []
    for i, ch in enumerate(chunks, start=1):
        label = str(ch.get("citation_label") or "Referência")
        body = str(ch.get("content") or "").strip()
        parts.append(f"[{i}] ({label})\n{body}")
    return "\n\n".join(parts)


def build_system_prompt(settings: Settings, *, context_block: str) -> str:
    return f"{settings.lexai_system_prompt}\n\nContexto normativo recuperado:\n{context_block}"


def build_gemini_turn_messages(turn_dicts: list[dict]):
    return build_gemini_turn_contents(turn_dicts)
