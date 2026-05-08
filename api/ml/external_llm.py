"""Enriquecimento opcional do parecer via API compatível com OpenAI (Chat Completions)."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

import httpx

logger = logging.getLogger(__name__)


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
    m = model.lower()
    if "gpt-4o-mini" in m or "4o-mini" in m:
        inp, out = 0.15, 0.60
    elif "gpt-4o" in m:
        inp, out = 2.50, 10.0
    elif "gpt-3.5" in m:
        inp, out = 0.50, 1.50
    else:
        inp, out = 0.15, 0.60
    return (input_tokens * inp + output_tokens * out) / 1_000_000


def _build_user_prompt(
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
        "Com base apenas nos trechos e resumos abaixo (não invente fatos ou números), "
        "produza um parágrafo único (máx. 120 palavras) com observações jurídicas "
        "preliminares para um advogado brasileiro: ângulos de revisão, cautelas e "
        "possíveis próximos passos processuais ou negociais.\n\n"
        f"Área inferida: {contract_type}. Risco: {risk_level}. Tipo de documento: {document_kind}.\n"
        f"Resumo executivo (modelo local): {executive_summary[:1200]}\n"
        f"Principais riscos (heurística): {risks}\n"
        f"Recomendações automáticas: {recs}\n"
        f"Trecho representativo do texto:\n{ex}\n"
    )


def maybe_enrich_opinion(
    *,
    mode: str,
    gate_triggered: bool,
    api_key: str | None,
    base_url: str,
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
        used=False,
        text="",
        model=None,
        input_tokens=None,
        output_tokens=None,
        cost_usd=0.0,
    )
    if not should_invoke_external_llm(mode, gate_triggered, api_key):
        return empty, time.perf_counter() - t0

    url = base_url.rstrip("/") + "/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    user_content = _build_user_prompt(
        executive_summary=executive_summary,
        main_risks=main_risks,
        recommendations=recommendations,
        contract_type=contract_type,
        risk_level=risk_level,
        document_kind=document_kind,
        excerpt=excerpt,
    )
    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Você apoia advogados no Brasil. Seja preciso, conservador e não invente "
                    "fatos ou citações. Se faltar contexto, diga que a revisão humana é necessária."
                ),
            },
            {"role": "user", "content": user_content},
        ],
        "max_tokens": 500,
        "temperature": 0.25,
    }

    try:
        with httpx.Client(timeout=timeout_sec) as client:
            response = client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
    except Exception as exc:
        logger.warning("Chamada ao LLM externo falhou: %s", exc)
        return empty, time.perf_counter() - t0

    try:
        choice0 = (data.get("choices") or [{}])[0]
        msg = (choice0.get("message") or {})
        content = str(msg.get("content") or "").strip()
    except (TypeError, IndexError, AttributeError):
        content = ""

    usage = data.get("usage") or {}
    in_tok = int(usage.get("prompt_tokens") or 0)
    out_tok = int(usage.get("completion_tokens") or 0)

    if not content:
        logger.warning("LLM externo retornou conteúdo vazio.")
        return empty, time.perf_counter() - t0

    cost = _estimate_cost_usd(model, in_tok, out_tok)
    elapsed = time.perf_counter() - t0
    return (
        ExternalEnrichmentResult(
            used=True,
            text=content,
            model=model,
            input_tokens=in_tok or None,
            output_tokens=out_tok or None,
            cost_usd=round(cost, 6),
        ),
        elapsed,
    )
