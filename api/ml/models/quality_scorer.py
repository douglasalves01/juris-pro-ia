"""Score de qualidade de peca/documento antes de protocolo."""

from __future__ import annotations

import re
from dataclasses import dataclass

from api.ml.document_kind import detect_document_kind
from api.ml.preprocessor import TextPreprocessor


@dataclass(frozen=True)
class QualitySuggestion:
    dimension: str
    issue: str
    fix: str


@dataclass(frozen=True)
class QualityResult:
    score: int
    dimensions: dict[str, int]
    suggestions: list[QualitySuggestion]


def _has_any(text: str, patterns: list[str]) -> bool:
    return any(re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE) for pattern in patterns)


def _score_completeness(text: str, document_kind: str, sections: dict[str, list[str]]) -> tuple[int, list[QualitySuggestion]]:
    suggestions: list[QualitySuggestion] = []
    if document_kind == "peticao_inicial":
        checks = {
            "partes": _has_any(text, [r"\bautor\b", r"\brequerente\b"]) and _has_any(text, [r"\br[eé]u\b", r"\brequerid[oa]\b"]),
            "fatos": _has_any(text, [r"\bDOS\s+FATOS\b", r"\bfatos\b"]),
            "direito": _has_any(text, [r"\bDO\s+DIREITO\b", r"\bfundamentos?\b"]),
            "pedidos": _has_any(text, [r"\bDOS\s+PEDIDOS\b", r"\brequer\b", r"\bpedido\b"]),
            "valor": _has_any(text, [r"valor\s+da\s+causa", r"R\$\s*\d"]),
        }
    elif document_kind == "contrato":
        checks = {
            "partes": bool(sections.get("partes")) or _has_any(text, [r"\bcontratante\b", r"\bcontratad[oa]\b"]),
            "objeto": _has_any(text, [r"\bobjeto\b", r"contrato\s+de"]),
            "preco": bool(sections.get("valores")) or _has_any(text, [r"\bpre[cç]o\b", r"\bpagamento\b"]),
            "prazo": _has_any(text, [r"\bprazo\b", r"\bvig[eê]ncia\b"]),
            "foro": _has_any(text, [r"\bforo\b", r"\bcompet[eê]ncia\b"]),
        }
    else:
        checks = {
            "identificacao": len(text.split()) >= 80,
            "fundamentacao": _has_any(text, [r"\bart\.?\s*\d", r"\blei\b", r"\bc[oó]digo\b"]),
            "pedidos_ou_objeto": _has_any(text, [r"\bpedido\b", r"\bobjeto\b", r"\brequer\b"]),
        }
    passed = sum(1 for ok in checks.values() if ok)
    score = round(100 * passed / max(1, len(checks)))
    for name, ok in checks.items():
        if not ok:
            suggestions.append(
                QualitySuggestion(
                    dimension="completeness",
                    issue=f"Elemento estrutural ausente ou fraco: {name}.",
                    fix=f"Incluir ou reforcar a secao/campo de {name} antes do protocolo.",
                )
            )
    return score, suggestions


def _score_coherence(text: str) -> tuple[int, list[QualitySuggestion]]:
    suggestions: list[QualitySuggestion] = []
    words = text.split()
    if len(words) < 40:
        return 35, [
            QualitySuggestion(
                dimension="coherence",
                issue="Texto curto demais para avaliar coerencia argumentativa.",
                fix="Desenvolver fatos, fundamentos e conclusao com encadeamento logico.",
            )
        ]
    connectors = len(re.findall(r"\b(portanto|assim|dessa forma|por isso|alem disso|contudo|entretanto|logo)\b", text, re.I))
    contradiction_markers = len(re.findall(r"\b(contraditoriamente|inconsistente|sem relacao|nao obstante)\b", text, re.I))
    score = min(100, 55 + connectors * 8 - contradiction_markers * 12)
    if connectors == 0:
        suggestions.append(
            QualitySuggestion(
                dimension="coherence",
                issue="Poucos conectores logicos entre fatos, fundamentos e conclusao.",
                fix="Adicionar transicoes argumentativas para explicitar a relacao entre premissas e pedidos.",
            )
        )
    return max(0, score), suggestions


def _score_citations(text: str) -> tuple[int, list[QualitySuggestion]]:
    suggestions: list[QualitySuggestion] = []
    citations = re.findall(r"\b(?:art\.?|artigo)\s*\d+|\blei\s+\d|CPC|CDC|CLT|LGPD|c[oó]digo civil", text, re.I)
    score = min(100, 30 + len(citations) * 18)
    if not citations:
        suggestions.append(
            QualitySuggestion(
                dimension="citations",
                issue="Nenhuma citacao normativa verificavel foi encontrada.",
                fix="Inserir fundamentos legais pertinentes e conferir a aderencia aos fatos.",
            )
        )
    return score, suggestions


def _score_language(text: str) -> tuple[int, list[QualitySuggestion]]:
    suggestions: list[QualitySuggestion] = []
    words = text.split()
    avg_word_len = sum(len(word) for word in words) / max(1, len(words))
    all_caps_ratio = sum(1 for word in words if len(word) > 3 and word.isupper()) / max(1, len(words))
    typo_markers = len(re.findall(r"\s{2,}|[!?]{2,}|[,;:]{2,}", text))
    score = 88
    if avg_word_len > 9:
        score -= 12
        suggestions.append(
            QualitySuggestion(
                dimension="language",
                issue="Linguagem possivelmente excessivamente densa.",
                fix="Revisar frases longas e reduzir termos pouco necessários.",
            )
        )
    if all_caps_ratio > 0.15:
        score -= 15
        suggestions.append(
            QualitySuggestion(
                dimension="language",
                issue="Uso elevado de palavras em caixa alta.",
                fix="Usar caixa alta apenas para titulos e siglas.",
            )
        )
    if typo_markers:
        score -= min(20, typo_markers * 4)
    return max(0, score), suggestions


def score(text: str) -> QualityResult:
    clean_text = TextPreprocessor().clean(text or "")
    sections = TextPreprocessor().extract_sections(clean_text)
    document_kind = detect_document_kind(clean_text)
    completeness, s1 = _score_completeness(clean_text, document_kind, sections)
    coherence, s2 = _score_coherence(clean_text)
    citations, s3 = _score_citations(clean_text)
    language, s4 = _score_language(clean_text)
    dimensions = {
        "completeness": completeness,
        "coherence": coherence,
        "citations": citations,
        "language": language,
    }
    final_score = round(
        completeness * 0.35
        + coherence * 0.25
        + citations * 0.25
        + language * 0.15
    )
    return QualityResult(
        score=max(0, min(100, final_score)),
        dimensions=dimensions,
        suggestions=(s1 + s2 + s3 + s4)[:12],
    )
