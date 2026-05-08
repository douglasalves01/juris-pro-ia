"""Geração de resumo jurídico em linguagem acessível."""

from __future__ import annotations

import base64
import re
from io import BytesIO

import httpx
from fpdf import FPDF


_LEGAL_TERMS = {
    "adimplemento": "cumprimento",
    "agravante": "quem recorre",
    "agravado": "parte contrária no recurso",
    "ajuizar": "entrar com",
    "aludido": "citado",
    "anuência": "concordância",
    "autor": "quem entrou com a ação",
    "comarca": "região do fórum",
    "consubstanciado": "baseado",
    "contestação": "defesa",
    "deferimento": "aprovação",
    "demandado": "parte processada",
    "demandante": "quem entrou com o pedido",
    "exequente": "quem cobra",
    "executado": "quem está sendo cobrado",
    "inadimplemento": "não pagamento",
    "indeferimento": "rejeição",
    "interposição": "apresentação",
    "lide": "conflito",
    "litigante": "parte do processo",
    "mérito": "ponto principal do caso",
    "ônus": "responsabilidade",
    "parte ré": "parte acusada",
    "petição inicial": "pedido inicial",
    "pleito": "pedido",
    "requerente": "quem faz o pedido",
    "requerido": "contra quem o pedido é feito",
    "réu": "parte acusada",
    "sucumbência": "perda no processo",
    "tutela de urgência": "decisão rápida e provisória",
}

_INTERMEDIATE_TERMS = {
    key: value
    for key, value in _LEGAL_TERMS.items()
    if key
    in {
        "exequente",
        "executado",
        "inadimplemento",
        "sucumbência",
        "tutela de urgência",
        "consubstanciado",
        "interposição",
        "lide",
        "ônus",
        "pleito",
    }
}


def _replace_terms(text: str, mapping: dict[str, str]) -> str:
    out = text
    for source, target in sorted(mapping.items(), key=lambda item: len(item[0]), reverse=True):
        out = re.sub(rf"\b{re.escape(source)}\b", target, out, flags=re.IGNORECASE)
    return out


def _shorten_long_sentences(text: str, max_words: int = 40) -> str:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    out: list[str] = []
    for sentence in sentences:
        words = sentence.split()
        if len(words) <= max_words:
            out.append(sentence)
            continue
        chunks = [" ".join(words[idx : idx + max_words]) for idx in range(0, len(words), max_words)]
        out.extend(chunk.rstrip(".,;:") + "." for chunk in chunks if chunk)
    return " ".join(part for part in out if part).strip()


def simplify_by_rules(text: str, level: str) -> str:
    source = (text or "").strip()
    if not source:
        return ""
    if level == "tecnico":
        return source
    mapping = _INTERMEDIATE_TERMS if level == "intermediario" else _LEGAL_TERMS
    simplified = _replace_terms(source, mapping)
    if level == "leigo":
        simplified = _shorten_long_sentences(simplified)
    return simplified.strip()


def rewrite_with_llm(
    text: str,
    level: str,
    api_key: str,
    base_url: str,
    model: str,
) -> str:
    level_description = {
        "leigo": "simples, clara e acessível para uma pessoa sem formação jurídica",
        "intermediario": "objetiva, com poucos termos técnicos e explicações curtas",
        "tecnico": "técnica, mantendo a terminologia jurídica",
    }.get(level, "simples")
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": f"Você explica documentos jurídicos em linguagem {level_description}.",
            },
            {
                "role": "user",
                "content": (
                    "Reescreva o resumo abaixo mantendo os fatos e limitando a 200 palavras.\n\n"
                    f"Texto: {text}"
                ),
            },
        ],
        "temperature": 0.2,
    }
    url = base_url.rstrip("/") + "/chat/completions"
    with httpx.Client(timeout=25.0) as client:
        response = client.post(url, headers={"Authorization": f"Bearer {api_key}"}, json=payload)
        response.raise_for_status()
    data = response.json()
    content = data["choices"][0]["message"]["content"]
    if not isinstance(content, str) or not content.strip():
        raise ValueError("LLM retornou resumo vazio.")
    return content.strip()


def generate_summary(
    executive_summary: str,
    level: str,
    api_key: str | None,
    base_url: str,
    model: str,
) -> str:
    if api_key and level in {"leigo", "intermediario"}:
        try:
            return rewrite_with_llm(executive_summary, level, api_key, base_url, model)
        except Exception:
            pass
    return simplify_by_rules(executive_summary, level)


def generate_pdf_base64(summary_text: str, brand_name: str = "JurisPro IA") -> str:
    text = (summary_text or "").strip()
    if not text:
        raise ValueError("summary_text não pode ser vazio.")
    brand = (brand_name or "JurisPro IA").strip()
    safe_text = text.encode("latin-1", errors="replace").decode("latin-1")
    safe_brand = brand.encode("latin-1", errors="replace").decode("latin-1")
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, safe_brand, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    pdf.set_font("Helvetica", size=12)
    pdf.multi_cell(0, 8, safe_text)
    pdf_bytes = bytes(pdf.output())
    buffer = BytesIO(pdf_bytes)
    return base64.b64encode(buffer.getvalue()).decode("ascii")
