"""Checklist de compliance por regulacao com regras deterministicas."""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import httpx


ComplianceStatus = Literal["ok", "fail", "warning", "na"]


@dataclass(frozen=True)
class ComplianceItemResult:
    id: str
    description: str
    status: ComplianceStatus
    evidence: str | None = None


@dataclass(frozen=True)
class ComplianceResult:
    regulation: str
    items: list[ComplianceItemResult]


def _checklists_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "data" / "checklists"


def _normalize(value: str) -> str:
    nfkd = unicodedata.normalize("NFKD", value or "")
    no_accents = "".join(ch for ch in nfkd if not unicodedata.combining(ch))
    return no_accents.casefold()


def _contains(text: str, phrase: str) -> bool:
    normalized_text = _normalize(text)
    normalized_phrase = _normalize(phrase).strip()
    if not normalized_phrase:
        return False
    return re.search(rf"(?<!\w){re.escape(normalized_phrase)}(?!\w)", normalized_text) is not None


def _first_evidence(text: str, phrases: list[str]) -> str | None:
    normalized_text = _normalize(text)
    for phrase in phrases:
        normalized_phrase = _normalize(phrase).strip()
        if not normalized_phrase:
            continue
        match = re.search(rf"(?<!\w){re.escape(normalized_phrase)}(?!\w)", normalized_text)
        if match:
            start = max(0, match.start() - 80)
            end = min(len(text), match.end() + 120)
            return re.sub(r"\s+", " ", text[start:end]).strip()
    return None


def load_checklists(base_dir: Path | None = None) -> list[dict]:
    directory = base_dir or _checklists_dir()
    checklists = []
    for path in sorted(directory.glob("*.json")):
        checklists.append(json.loads(path.read_text(encoding="utf-8")))
    return checklists


def select_regulations(
    text: str,
    contract_type: str = "",
    document_kind: str = "",
    max_regulations: int = 2,
) -> list[dict]:
    checklists = load_checklists()
    haystack = " ".join([text or "", contract_type or "", document_kind or ""])
    scored: list[tuple[int, dict]] = []
    for checklist in checklists:
        keywords = [str(k) for k in checklist.get("area_keywords") or []]
        score = sum(1 for keyword in keywords if _contains(haystack, keyword))
        regulation = str(checklist.get("regulation") or "").upper()
        if regulation == "CPC" and document_kind in {"peticao_inicial", "contestacao", "recurso"}:
            score += 2
        if regulation == "CDC" and "consumidor" in _normalize(contract_type):
            score += 2
        if regulation == "CLT" and "trabalh" in _normalize(contract_type):
            score += 2
        if score > 0:
            scored.append((score, checklist))

    if not scored:
        return []
    scored.sort(key=lambda item: (-item[0], str(item[1].get("regulation") or "")))
    return [item[1] for item in scored[:max_regulations]]


def _classify_item(text: str, item: dict) -> ComplianceItemResult:
    required = [str(v) for v in item.get("required") or []]
    optional = [str(v) for v in item.get("optional") or []]
    required_hits = [phrase for phrase in required if _contains(text, phrase)]
    optional_hits = [phrase for phrase in optional if _contains(text, phrase)]

    if required_hits:
        status: ComplianceStatus = "ok"
        evidence = _first_evidence(text, required_hits)
    elif optional_hits:
        status = "warning"
        evidence = _first_evidence(text, optional_hits)
    else:
        status = "fail"
        evidence = None

    return ComplianceItemResult(
        id=str(item.get("id") or "item"),
        description=str(item.get("description") or ""),
        status=status,
        evidence=evidence,
    )


def _verify_ambiguous_with_llm(
    text: str,
    result: ComplianceResult,
    api_key: str | None,
    base_url: str,
    model: str,
) -> ComplianceResult:
    if not api_key or not any(item.status == "warning" for item in result.items):
        return result

    prompt_items = [
        {"id": item.id, "description": item.description}
        for item in result.items
        if item.status == "warning"
    ]
    prompt = (
        "Voce verifica compliance juridico. "
        "Para cada item, responda uma lista JSON com id, status "
        '("ok", "fail", "warning" ou "na") e evidence curta.\n'
        f"Regulacao: {result.regulation}\nItens: {json.dumps(prompt_items, ensure_ascii=False)}\n"
        f"Texto: {text[:3000]}"
    )
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"maxOutputTokens": 800, "temperature": 0.0},
    }
    try:
        response = httpx.post(url, json=payload, timeout=20.0)
        response.raise_for_status()
        content = response.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
        # Remove possível markdown code block
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        parsed = json.loads(content)
    except Exception:
        return result

    overrides: dict[str, tuple[ComplianceStatus, str | None]] = {}
    if isinstance(parsed, list):
        for item in parsed:
            if not isinstance(item, dict):
                continue
            status = str(item.get("status") or "").lower()
            if status not in {"ok", "fail", "warning", "na"}:
                continue
            overrides[str(item.get("id") or "")] = (
                status,  # type: ignore[arg-type]
                str(item.get("evidence")) if item.get("evidence") else None,
            )

    if not overrides:
        return result
    return ComplianceResult(
        regulation=result.regulation,
        items=[
            ComplianceItemResult(
                id=item.id,
                description=item.description,
                status=overrides.get(item.id, (item.status, item.evidence))[0],
                evidence=overrides.get(item.id, (item.status, item.evidence))[1],
            )
            for item in result.items
        ],
    )


def check(
    text: str,
    contract_type: str = "",
    document_kind: str = "",
    api_key: str | None = None,
    base_url: str = "",  # não usado no Gemini, mantido por compatibilidade
    model: str = "gemini-2.0-flash",
) -> list[ComplianceResult]:
    selected = select_regulations(text, contract_type, document_kind)
    results: list[ComplianceResult] = []
    for checklist in selected:
        items = [_classify_item(text, item) for item in checklist.get("items") or [] if isinstance(item, dict)]
        result = ComplianceResult(regulation=str(checklist.get("regulation") or ""), items=items)
        results.append(_verify_ambiguous_with_llm(text, result, api_key, base_url, model))
    return results
