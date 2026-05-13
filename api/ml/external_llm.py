"""Enriquecimento opcional do parecer via Google Gemini API."""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any

import httpx

logger = logging.getLogger(__name__)

# Endpoint Gemini generateContent
_GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta/models"
_RETRYABLE_STATUS_CODES = {408, 409, 429, 500, 502, 503, 504}


@dataclass(frozen=True)
class ExternalEnrichmentResult:
    used: bool
    text: str
    model: str | None
    input_tokens: int | None
    output_tokens: int | None
    cost_usd: float


@dataclass(frozen=True)
class AnalysisFallbackResult:
    used: bool
    executive_summary: str | None
    main_risks: list[str]
    recommendations: list[str]
    positive_points: list[str]
    outcome_rationale: str | None
    outcome_probability: float | None
    outcome_confidence: float | None
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


def _build_fallback_prompt(
    *,
    missing_fields: list[str],
    executive_summary: str,
    main_risks: list[str],
    recommendations: list[str],
    positive_points: list[str],
    contract_type: str,
    risk_level: str,
    risk_score: int,
    win_prediction: str,
    win_probability: float,
    document_kind: str,
    excerpt: str,
) -> str:
    ex = (excerpt or "").strip()[:7000]
    return (
        "Você apoia advogados no Brasil. Complete somente os campos solicitados, "
        "com base no texto fornecido. Seja conservador, não invente fatos, não cite "
        "jurisprudência inexistente e indique incerteza quando necessário.\n\n"
        "Retorne exclusivamente JSON válido, sem markdown, no formato:\n"
        "{"
        '"executive_summary":"string",'
        '"main_risks":["string"],'
        '"recommendations":["string"],'
        '"positive_points":["string"],'
        '"outcome_probability":{"value":0.0,"rationale":"string","confidence":0.0}'
        "}\n\n"
        f"Campos que precisam de fallback: {', '.join(missing_fields)}.\n"
        f"Tipo de documento: {document_kind}. Área inferida: {contract_type}. "
        f"Risco local: {risk_level} ({risk_score}/100). "
        f"Previsão local: {win_prediction} ({win_probability}).\n"
        f"Resumo local: {executive_summary[:1200]}\n"
        f"Riscos locais: {'; '.join(main_risks[:8])}\n"
        f"Recomendações locais: {'; '.join(recommendations[:8])}\n"
        f"Pontos positivos locais: {'; '.join(positive_points[:6])}\n"
        f"Texto:\n{ex}\n"
    )


def _extract_json_object(text: str) -> dict[str, Any]:
    raw = (text or "").strip()
    if not raw:
        return {}
    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", raw, re.DOTALL)
    if fenced:
        raw = fenced.group(1)
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        for index, char in enumerate(raw):
            if char != "{":
                continue
            try:
                parsed, _ = decoder.raw_decode(raw[index:])
            except json.JSONDecodeError:
                continue
            return parsed if isinstance(parsed, dict) else {}
        logger.warning("Gemini fallback retornou JSON inválido.")
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _extract_json_array(text: str) -> list[Any]:
    raw = (text or "").strip()
    if not raw:
        return []
    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", raw, re.DOTALL)
    if fenced:
        raw = fenced.group(1)
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        for index, char in enumerate(raw):
            if char != "[":
                continue
            try:
                parsed, _ = decoder.raw_decode(raw[index:])
            except json.JSONDecodeError:
                continue
            return parsed if isinstance(parsed, list) else []
        return []
    return parsed if isinstance(parsed, list) else []


def _string_list(value: Any, *, limit: int) -> list[str]:
    if not isinstance(value, list):
        return []
    out = [str(item).strip() for item in value if str(item).strip()]
    return out[:limit]


def _bounded_probability(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return max(0.0, min(1.0, parsed))


def _clean_model_text(text: str, *, limit: int = 700) -> str:
    cleaned = re.sub(r"\s+", " ", (text or "").strip())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 1].rstrip() + "…"


def _textual_fallback_result(content: str) -> dict[str, Any]:
    text = _clean_model_text(content)
    if not text:
        return {}
    return {
        "main_risks": [text],
        "recommendations": ["Revisar os riscos apontados pelo fallback Gemini com validação jurídica humana."],
        "positive_points": ["Fallback Gemini retornou análise textual complementar."],
        "outcome_probability": {
            "value": 0.5,
            "rationale": text,
            "confidence": 0.4,
        },
    }


def _retry_after_seconds(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        pass
    try:
        parsed = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return max(0.0, (parsed - datetime.now(timezone.utc)).total_seconds())


def _retry_delay(attempt: int, response: httpx.Response | None, base_delay_sec: float) -> float:
    retry_after = _retry_after_seconds(response.headers.get("retry-after") if response else None)
    if retry_after is not None:
        return min(retry_after, 30.0)
    return min(base_delay_sec * (2 ** attempt), 12.0)


def _call_gemini(
    prompt: str,
    api_key: str,
    model: str,
    timeout_sec: float,
    *,
    response_mime_type: str | None = None,
    response_schema: dict[str, Any] | None = None,
    max_attempts: int = 3,
    retry_base_delay_sec: float = 1.5,
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
    if response_mime_type:
        payload["generationConfig"]["responseMimeType"] = response_mime_type
    if response_schema:
        payload["generationConfig"]["responseSchema"] = response_schema

    attempts = max(1, max_attempts)
    last_exc: Exception | None = None
    with httpx.Client(timeout=timeout_sec) as client:
        for attempt in range(attempts):
            response: httpx.Response | None = None
            try:
                response = client.post(url, json=payload)
                if response.status_code in _RETRYABLE_STATUS_CODES and attempt < attempts - 1:
                    delay = _retry_delay(attempt, response, retry_base_delay_sec)
                    logger.warning(
                        "Gemini retornou HTTP %s; retry %s/%s em %.1fs.",
                        response.status_code,
                        attempt + 2,
                        attempts,
                        delay,
                    )
                    time.sleep(delay)
                    continue
                response.raise_for_status()
                data = response.json()
                break
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_exc = exc
                if attempt >= attempts - 1:
                    raise
                delay = _retry_delay(attempt, response, retry_base_delay_sec)
                logger.warning(
                    "Falha transitória ao chamar Gemini; retry %s/%s em %.1fs: %s",
                    attempt + 2,
                    attempts,
                    delay,
                    exc,
                )
                time.sleep(delay)
            except httpx.HTTPStatusError as exc:
                last_exc = exc
                status_code = exc.response.status_code
                if status_code not in _RETRYABLE_STATUS_CODES or attempt >= attempts - 1:
                    raise
                delay = _retry_delay(attempt, exc.response, retry_base_delay_sec)
                logger.warning(
                    "Gemini retornou HTTP %s; retry %s/%s em %.1fs.",
                    status_code,
                    attempt + 2,
                    attempts,
                    delay,
                )
                time.sleep(delay)
        else:
            if last_exc is not None:
                raise last_exc
            raise RuntimeError("Gemini não retornou resposta.")

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


def _fallback_response_schema() -> dict[str, Any]:
    return {
        "type": "OBJECT",
        "properties": {
            "executive_summary": {"type": "STRING"},
            "main_risks": {"type": "ARRAY", "items": {"type": "STRING"}},
            "recommendations": {"type": "ARRAY", "items": {"type": "STRING"}},
            "positive_points": {"type": "ARRAY", "items": {"type": "STRING"}},
            "outcome_probability": {
                "type": "OBJECT",
                "properties": {
                    "value": {"type": "NUMBER"},
                    "rationale": {"type": "STRING"},
                    "confidence": {"type": "NUMBER"},
                },
                "required": ["value", "rationale", "confidence"],
            },
        },
        "required": [
            "executive_summary",
            "main_risks",
            "recommendations",
            "positive_points",
            "outcome_probability",
        ],
    }


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


def complete_analysis_fallback(
    *,
    api_key: str | None,
    model: str,
    missing_fields: list[str],
    executive_summary: str,
    main_risks: list[str],
    recommendations: list[str],
    positive_points: list[str],
    contract_type: str,
    risk_level: str,
    risk_score: int,
    win_prediction: str,
    win_probability: float,
    document_kind: str,
    excerpt: str,
    timeout_sec: float = 60.0,
) -> tuple[AnalysisFallbackResult, float]:
    """Completa campos obrigatórios quando os modelos locais retornam saída parcial."""
    t0 = time.perf_counter()
    empty = AnalysisFallbackResult(
        used=False,
        executive_summary=None,
        main_risks=[],
        recommendations=[],
        positive_points=[],
        outcome_rationale=None,
        outcome_probability=None,
        outcome_confidence=None,
        model=None,
        input_tokens=None,
        output_tokens=None,
        cost_usd=0.0,
    )
    if not missing_fields or not (api_key and api_key.strip()):
        return empty, time.perf_counter() - t0

    prompt = _build_fallback_prompt(
        missing_fields=missing_fields,
        executive_summary=executive_summary,
        main_risks=main_risks,
        recommendations=recommendations,
        positive_points=positive_points,
        contract_type=contract_type,
        risk_level=risk_level,
        risk_score=risk_score,
        win_prediction=win_prediction,
        win_probability=win_probability,
        document_kind=document_kind,
        excerpt=excerpt,
    )
    try:
        content, in_tok, out_tok = _call_gemini(
            prompt,
            api_key,
            model,
            timeout_sec,
            response_mime_type="application/json",
            response_schema=_fallback_response_schema(),
        )
    except Exception as exc:
        logger.warning("Fallback Gemini falhou: %s", exc)
        return empty, time.perf_counter() - t0

    data = _extract_json_object(content)
    if not data:
        array_data = _extract_json_array(content)
        if array_data and isinstance(array_data[0], dict):
            data = array_data[0]
    if not data:
        logger.warning(
            "Gemini fallback retornou JSON inválido; usando fallback textual. raw=%r",
            _clean_model_text(content, limit=500),
        )
        data = _textual_fallback_result(content)
    if not data:
        return empty, time.perf_counter() - t0

    outcome = data.get("outcome_probability") if isinstance(data.get("outcome_probability"), dict) else {}
    cost = _estimate_cost_usd(model, in_tok, out_tok)
    return (
        AnalysisFallbackResult(
            used=True,
            executive_summary=str(data.get("executive_summary") or "").strip() or None,
            main_risks=_string_list(data.get("main_risks"), limit=10),
            recommendations=_string_list(data.get("recommendations"), limit=10),
            positive_points=_string_list(data.get("positive_points"), limit=6),
            outcome_rationale=str(outcome.get("rationale") or "").strip() or None,
            outcome_probability=_bounded_probability(outcome.get("value")),
            outcome_confidence=_bounded_probability(outcome.get("confidence")),
            model=model,
            input_tokens=in_tok or None,
            output_tokens=out_tok or None,
            cost_usd=round(cost, 6),
        ),
        time.perf_counter() - t0,
    )
