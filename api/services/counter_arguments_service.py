"""Geração de argumentos adversariais para revisão jurídica."""

from __future__ import annotations

import json
from typing import Literal, TypedDict

import httpx


class CounterArgument(TypedDict):
    text: str
    strength: Literal["forte", "medio", "fraco"]
    category: str


_SEVERITY_TO_STRENGTH = {
    "critical": "forte",
    "high": "forte",
    "crítica": "forte",
    "critica": "forte",
    "alta": "forte",
    "alto": "forte",
    "medium": "medio",
    "média": "medio",
    "media": "medio",
    "médio": "medio",
    "medio": "medio",
    "low": "fraco",
    "baixa": "fraco",
    "baixo": "fraco",
}

_TYPE_TO_CATEGORY = {
    "lgpd_compliance": "Proteção de Dados",
    "penalty_clause": "Cláusula Penal",
    "termination_clause": "Rescisão",
    "jurisdiction_clause": "Competência",
    "liability_limitation": "Responsabilidade",
    "intellectual_property": "Propriedade Intelectual",
    "dispute_resolution": "Resolução de Conflitos",
}


def _strength(value: str) -> Literal["forte", "medio", "fraco"]:
    mapped = _SEVERITY_TO_STRENGTH.get((value or "").strip().lower(), "medio")
    return mapped  # type: ignore[return-value]


def _category(value: str) -> str:
    return _TYPE_TO_CATEGORY.get((value or "").strip(), "Geral")


def build_from_attention_points(
    attention_points: list[dict],
    max_arguments: int,
) -> list[CounterArgument]:
    out: list[CounterArgument] = []
    for point in attention_points[:max_arguments]:
        if not isinstance(point, dict):
            continue
        description = str(point.get("description") or point.get("descricao") or "").strip()
        clause = str(point.get("clause") or point.get("tipo") or "").strip()
        if not description and not clause:
            continue
        argument_text = (
            "A parte contrária pode sustentar que "
            f"{description[:1].lower() + description[1:] if description else 'há fragilidade no ponto indicado'}"
        ).rstrip(".")
        out.append(
            {
                "text": argument_text + ".",
                "strength": _strength(str(point.get("severity") or point.get("severidade") or "")),
                "category": _category(clause),
            }
        )
    return out[:max_arguments]


def _coerce_arguments(items: object, max_arguments: int) -> list[CounterArgument]:
    if not isinstance(items, list):
        return []
    out: list[CounterArgument] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text") or item.get("argumento") or "").strip()
        strength = str(item.get("strength") or item.get("forca") or "medio").strip().lower()
        category = str(item.get("category") or item.get("categoria") or "Geral").strip()
        if not text:
            continue
        if strength not in {"forte", "medio", "fraco"}:
            strength = "medio"
        out.append({"text": text, "strength": strength, "category": category or "Geral"})  # type: ignore[list-item]
        if len(out) >= max_arguments:
            break
    return out


def build_with_llm(
    text: str,
    max_arguments: int,
    api_key: str,
    base_url: str,
    model: str,
) -> list[CounterArgument]:
    prompt = (
        f"Você é um advogado adversário experiente. "
        f"Liste até {max_arguments} argumentos que a parte contrária poderia usar. "
        'Responda apenas com uma lista JSON com campos "text", "strength" ("forte", "medio" ou "fraco") '
        f'e "category".\n\nTexto: {text[:3000]}'
    )
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"maxOutputTokens": 800, "temperature": 0.2},
    }
    with httpx.Client(timeout=25.0) as client:
        response = client.post(url, json=payload)
        response.raise_for_status()
    data = response.json()
    try:
        content = data["candidates"][0]["content"]["parts"][0]["text"].strip()
    except (KeyError, IndexError, TypeError):
        content = "[]"
    # Remove possível markdown code block
    if content.startswith("```"):
        content = content.split("```")[1]
        if content.startswith("json"):
            content = content[4:]
    parsed = json.loads(content)
    return _coerce_arguments(parsed, max_arguments)


def generate_counter_arguments(
    text: str,
    attention_points: list[dict],
    max_arguments: int,
    api_key: str | None,
    base_url: str,
    model: str,
) -> list[CounterArgument]:
    if api_key:
        try:
            generated = build_with_llm(text, max_arguments, api_key, base_url, model)
            if generated:
                return generated
        except Exception:
            pass
    return build_from_attention_points(attention_points, max_arguments)
