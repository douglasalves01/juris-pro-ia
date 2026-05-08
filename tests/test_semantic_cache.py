from __future__ import annotations

from api.services import semantic_cache


def test_semantic_cache_returns_similar_entry(monkeypatch) -> None:
    semantic_cache.clear()
    monkeypatch.setenv("JURISPRO_SEMANTIC_CACHE_THRESHOLD", "0.5")
    semantic_cache.put(
        "k1",
        "deep",
        "Consumidor",
        "plano de saude cobertura tratamento oncologico",
        "Parecer em cache.",
    )

    hit = semantic_cache.get("deep", "Consumidor", "cobertura de tratamento oncologico por plano de saude")

    assert hit == "Parecer em cache."
    assert semantic_cache.stats()["cacheHitRate"] == 1.0


def test_semantic_cache_miss_for_different_mode() -> None:
    semantic_cache.clear()
    semantic_cache.put("k1", "deep", "Consumidor", "texto igual", "Cache")

    assert semantic_cache.get("standard", "Consumidor", "texto igual") is None
    assert semantic_cache.stats()["cacheMisses"] == 1
