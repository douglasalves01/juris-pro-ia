from __future__ import annotations

from api.ml.external_llm import should_invoke_external_llm


def test_fast_never_invokes() -> None:
    assert not should_invoke_external_llm("fast", True, "sk-xxx")
    assert not should_invoke_external_llm("fast", False, None)


def test_standard_requires_gate_and_key() -> None:
    assert not should_invoke_external_llm("standard", False, "sk-xxx")
    assert not should_invoke_external_llm("standard", True, None)
    assert not should_invoke_external_llm("standard", True, "   ")
    assert should_invoke_external_llm("standard", True, "sk-xxx")


def test_deep_invokes_with_key_only() -> None:
    assert not should_invoke_external_llm("deep", False, None)
    assert should_invoke_external_llm("deep", False, "sk-xxx")
