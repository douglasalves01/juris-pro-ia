"""Happy-path unit tests for LexAI RAG prompt utilities (no external APIs)."""

from __future__ import annotations

from lexai.services.rag_prompt import build_context_block, last_turn_messages
from lexai.services.vector_rag import RagChunk, citations_from_chunks


def test_rag_prompt_happy_path() -> None:
    history: list[dict] = []
    for i in range(8):
        history.append({"role": "user", "content": f"u{i}", "parts": None})
        history.append({"role": "assistant", "content": f"a{i}", "parts": None})
    history.append({"role": "user", "content": "final", "parts": None})

    trimmed = last_turn_messages(history, max_turns=6)
    assert len(trimmed) == 12
    assert trimmed[-1]["content"] == "final"

    block = build_context_block(
        [{"citation_label": "Art. 421 — CC/2002", "content": "Disposição aplicável ao caso."}],
    )
    assert "Art. 421" in block
    assert "CC/2002" in block

    cites = citations_from_chunks(
        [RagChunk(id=7, content="Trecho curto", citation_label="Art. 421 — CC/2002")],
    )
    assert cites[0]["chunk_id"] == 7
    assert cites[0]["label"] == "Art. 421 — CC/2002"
    assert "Trecho" in cites[0]["snippet"]
