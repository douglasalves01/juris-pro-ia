"""Geracao de minutas juridicas com LLM opcional e fallback por templates."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx

from api.services.private_knowledge import few_shot_context


_DISCLAIMER = "Minuta gerada para apoio. Revisao humana por profissional habilitado e obrigatoria antes do uso."


def _templates_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "data" / "templates"


def _template_for(document_type: str) -> str:
    path = _templates_dir() / f"{document_type}.txt"
    if not path.is_file():
        path = _templates_dir() / "contrato.txt"
    return path.read_text(encoding="utf-8")


def _context_value(context: dict[str, Any], key: str) -> str:
    value = context.get(key)
    if isinstance(value, list):
        return "\n".join(f"- {item}" for item in value if str(item).strip()) or "[REVISAR]"
    if isinstance(value, dict):
        return "\n".join(f"- {k}: {v}" for k, v in value.items()) or "[REVISAR]"
    return str(value).strip() if value else "[REVISAR]"


def build_with_template(
    document_type: str,
    context: dict[str, Any],
    style: str = "formal",
) -> dict[str, Any]:
    template = _template_for(document_type)
    values = {
        "parties": _context_value(context, "parties"),
        "subject": _context_value(context, "subject"),
        "facts": _context_value(context, "facts"),
        "claims": _context_value(context, "claims"),
    }
    draft = template.format(**values)
    if style == "conciso":
        draft = "\n".join(line for line in draft.splitlines() if line.strip())
    private_examples = context.get("privateExamples")
    if isinstance(private_examples, list) and private_examples:
        refs = "\n".join(
            f"- {item.get('title', 'Referencia')}: {item.get('text', '')[:300]}"
            for item in private_examples
            if isinstance(item, dict)
        )
        if refs:
            draft += "\n\nREFERENCIAS PRIVADAS DO ESCRITORIO\n" + refs
    sections = []
    current_title = "Minuta"
    current_lines: list[str] = []
    for line in draft.splitlines():
        stripped = line.strip()
        is_heading = stripped.isupper() and len(stripped.split()) <= 6 and "[REVISAR]" not in stripped
        if is_heading:
            if current_lines:
                content = "\n".join(current_lines).strip()
                sections.append(
                    {
                        "title": current_title,
                        "content": content,
                        "needsReview": "[REVISAR]" in content,
                        "confidence": 0.55 if "[REVISAR]" in content else 0.72,
                    }
                )
            current_title = stripped.title()
            current_lines = []
        else:
            current_lines.append(line)
    if current_lines:
        content = "\n".join(current_lines).strip()
        sections.append(
            {
                "title": current_title,
                "content": content,
                "needsReview": "[REVISAR]" in content,
                "confidence": 0.55 if "[REVISAR]" in content else 0.72,
            }
        )
    return {"draft": draft, "sections": sections, "disclaimer": _DISCLAIMER}


def build_with_llm(
    document_type: str,
    context: dict[str, Any],
    style: str,
    api_key: str,
    base_url: str,
    model: str,
) -> dict[str, Any]:
    prompt = (
        "Voce redige minutas juridicas em portugues brasileiro. "
        "Responda apenas JSON com draft, sections e disclaimer.\n\n"
        f"Tipo: {document_type}\nEstilo: {style}\n"
        f"Contexto: {json.dumps(context, ensure_ascii=False)}\n"
        "Marque trechos de baixa confianca com [REVISAR]."
    )
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"maxOutputTokens": 2000, "temperature": 0.2},
    }
    response = httpx.post(url, json=payload, timeout=35.0)
    response.raise_for_status()
    data = response.json()
    try:
        content = data["candidates"][0]["content"]["parts"][0]["text"].strip()
    except (KeyError, IndexError, TypeError):
        content = ""
    # Remove possível markdown code block
    if content.startswith("```"):
        content = content.split("```")[1]
        if content.startswith("json"):
            content = content[4:]
    parsed = json.loads(content)
    if not isinstance(parsed, dict) or not parsed.get("draft"):
        raise ValueError("Gemini retornou minuta invalida.")
    parsed.setdefault("sections", [])
    parsed["disclaimer"] = parsed.get("disclaimer") or _DISCLAIMER
    return parsed


def generate_draft(
    document_type: str,
    context: dict[str, Any],
    style: str,
    api_key: str | None,
    base_url: str,
    model: str,
    firm_id: str | None = None,
) -> dict[str, Any]:
    if firm_id:
        query = json.dumps(context, ensure_ascii=False)
        examples = few_shot_context(firm_id, f"{document_type}\n{query}", top_k=3)
        if examples:
            context = {**context, "privateExamples": examples}
    if api_key:
        try:
            return build_with_llm(document_type, context, style, api_key, base_url, model)
        except Exception:
            pass
    return build_with_template(document_type, context, style)
