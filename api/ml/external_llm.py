"""Enriquecimento opcional do parecer via Google Gemini API."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

import httpx

logger = logging.getLogger(__name__)

# Endpoint Gemini generateContent
_GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta/models"


@dataclass(frozen=True)
class ExternalEnrichmentResult:
    used: bool
    text: str
    model: str | None
    input_tokens: int | None
    output_tokens: int | None
    cost_usd: float


def should_invoke_external_llm(mode: str, gate_triggered: bool, api_key: str | None) -> bool:
    if mode == "fast" or not (api_key and api_key.strip()):
        return False
    if mode == "deep":
        return True
    return bool(gate_triggered)


def _estimate_cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    """Estimativa de custo baseada nos preços públicos do Gemini (mai/2026)."""
    m = model.lower()
    if "flash" in m:
        # gemini-2.0-flash / gemini-1.5-flash
        inp, out = 0.075, 0.30
    elif "pro" in m:
        # gemini-1.5-pro
        inp, out = 1.25, 5.00
    else:
        inp, out = 0.075, 0.30
    return (input_tokens * inp + output_tokens * out) / 1_000_000


def _build_prompt(
    *,
    executive_summary: str,
    main_risks: list[str],
    recommendations: list[str],
    contract_type: str,
    risk_level: str,
    document_kind: str,
    excerpt: str,
) -> str:
    risks = "; ".join(main_risks[:6]) if main_risks else "(nenhum)"
    recs = "; ".join(recommendations[:6]) if recommendations else "(nenhum)"
    ex = (excerpt or "").strip()[:4000]
    return (
        "Você apoia advogados no Brasil. Seja preciso, conservador e não invente "
        "fatos ou citações. Se faltar contexto, diga que a revisão humana é necessária.\n\n"
        "Com base apenas nos trechos e resumos abaixo, produza um parágrafo único "
        "(máx. 120 palavras) com observações jurídicas preliminares para um advogado "
        "brasileiro: ângulos de revisão, cautelas e possíveis próximos passos "
        "processuais ou negociais.\n\n"
        f"Área inferida: {contract_type}. Risco: {risk_level}. Tipo de documento: {document_kind}.\n"
        f"Resumo executivo (modelo local): {executive_summary[:1200]}\n"
        f"Principais riscos (heurística): {risks}\n"
        f"Recomendações automáticas: {recs}\n"
        f"Trecho representativo do texto:\n{ex}\n"
    )


def _call_gemini(
    prompt: str,
    api_key: str,
    model: str,
    timeout_sec: float,
) -> tuple[str, int, int]:
    """Chama a API Gemini e retorna (texto, input_tokens, output_tokens)."""
    url = f"{_GEMINI_BASE}/{model}:generateContent?key={api_key}"
    payload: dict[str, Any] = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "maxOutputTokens": 500,
            "temperature": 0.25,
        },
    }
    with httpx.Client(timeout=timeout_sec) as client:
        response = client.post(url, json=payload)
        response.raise_for_status()
        data = response.json()

    # Extrai texto
    try:
        content = data["candidates"][0]["content"]["parts"][0]["text"].strip()
    except (KeyError, IndexError, TypeError):
        content = ""

    # Extrai contagem de tokens (usageMetadata)
    usage = data.get("usageMetadata") or {}
    in_tok = int(usage.get("promptTokenCount") or 0)
    out_tok = int(usage.get("candidatesTokenCount") or 0)

    return content, in_tok, out_tok


def maybe_enrich_opinion(
    *,
    mode: str,
    gate_triggered: bool,
    api_key: str | None,
    base_url: str,  # mantido na assinatura por compatibilidade, não usado no Gemini
    model: str,
    executive_summary: str,
    main_risks: list[str],
    recommendations: list[str],
    contract_type: str,
    risk_level: str,
    document_kind: str,
    excerpt: str,
    timeout_sec: float = 60.0,
) -> tuple[ExternalEnrichmentResult, float]:
    """Retorna (resultado, duração em segundos). Falhas não levantam exceção."""
    t0 = time.perf_counter()
    empty = ExternalEnrichmentResult(
        used=False, text="", model=None,
        input_tokens=None, output_tokens=None, cost_usd=0.0,
    )

    if not should_invoke_external_llm(mode, gate_triggered, api_key):
        return empty, time.perf_counter() - t0

    prompt = _build_prompt(
        executive_summary=executive_summary,
        main_risks=main_risks,
        recommendations=recommendations,
        contract_type=contract_type,
        risk_level=risk_level,
        document_kind=document_kind,
        excerpt=excerpt,
    )

    try:
        content, in_tok, out_tok = _call_gemini(prompt, api_key, model, timeout_sec)
    except Exception as exc:
        logger.warning("Chamada ao Gemini falhou: %s", exc)
        return empty, time.perf_counter() - t0

    if not content:
        logger.warning("Gemini retornou conteúdo vazio.")
        return empty, time.perf_counter() - t0

    cost = _estimate_cost_usd(model, in_tok, out_tok)
    return (
        ExternalEnrichmentResult(
            used=True,
            text=content,
            model=model,
            input_tokens=in_tok or None,
            output_tokens=out_tok or None,
            cost_usd=round(cost, 6),
        ),
        time.perf_counter() - t0,
    )
