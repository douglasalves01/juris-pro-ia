"""Classificação heurística de urgência processual."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from datetime import date, datetime, timezone


@dataclass(frozen=True)
class UrgencyResult:
    score: int
    level: str
    rationale: str


_KEYWORDS_IMEDIATO = frozenset(
    [
        "liminar",
        "tutela de urgência",
        "tutela antecipada",
        "prazo fatal",
        "audiência",
        "penhora",
        "arresto",
        "sequestro",
        "busca e apreensão",
    ]
)

_KEYWORDS_URGENTE = frozenset(
    [
        "prazo",
        "intimação",
        "citação",
        "recurso",
        "apelação",
        "embargos",
        "contestação",
        "impugnação",
        "notificação",
    ]
)

_MONTHS = {
    "janeiro": 1,
    "fevereiro": 2,
    "marco": 3,
    "março": 3,
    "abril": 4,
    "maio": 5,
    "junho": 6,
    "julho": 7,
    "agosto": 8,
    "setembro": 9,
    "outubro": 10,
    "novembro": 11,
    "dezembro": 12,
}


def _strip_accents(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    return "".join(ch for ch in normalized if not unicodedata.combining(ch))


def _normalize_text(value: str) -> str:
    return _strip_accents(value or "").casefold()


def _contains_keyword(text: str, keywords: frozenset[str]) -> str | None:
    normalized = _normalize_text(text)
    for keyword in sorted(keywords, key=len, reverse=True):
        pattern = rf"(?<!\w){re.escape(_normalize_text(keyword))}(?!\w)"
        if re.search(pattern, normalized):
            return keyword
    return None


def _parse_date(date_str: str) -> date | None:
    value = (date_str or "").strip()
    if not value:
        return None

    for fmt in ("%d/%m/%Y", "%d-%m-%Y", "%d/%m/%y", "%d-%m-%y"):
        try:
            return datetime.strptime(value, fmt).date()
        except ValueError:
            pass

    month_year = re.fullmatch(r"\s*(\d{1,2})[/-](\d{4})\s*", value)
    if month_year:
        month = int(month_year.group(1))
        year = int(month_year.group(2))
        if 1 <= month <= 12:
            return date(year, month, 1)

    month_name = re.fullmatch(
        r"\s*(\d{1,2})\s+de\s+([A-Za-zÀ-ÿçÇ]+)\s+de\s+(\d{4})\s*",
        value,
        flags=re.IGNORECASE,
    )
    if month_name:
        day = int(month_name.group(1))
        month = _MONTHS.get(_normalize_text(month_name.group(2)))
        year = int(month_name.group(3))
        if month is not None:
            try:
                return date(year, month, day)
            except ValueError:
                return None

    return None


def _days_until(d: date) -> int:
    return (d - datetime.now(timezone.utc).date()).days


def classify(
    text: str,
    entities_datas: list[str],
    contract_type: str = "",
) -> UrgencyResult:
    body = text or ""
    parsed_dates = [(raw, _parse_date(raw)) for raw in entities_datas or []]
    dated = [(raw, parsed) for raw, parsed in parsed_dates if parsed is not None]
    if dated:
        closest = min(dated, key=lambda item: abs(_days_until(item[1])))
        for raw, parsed in dated:
            days = _days_until(parsed)
            if 0 <= days <= 7:
                return UrgencyResult(
                    score=90,
                    level="IMEDIATO",
                    rationale=f"Data próxima detectada ({raw}) em {days} dia(s), exigindo prioridade imediata.",
                )
        for raw, parsed in dated:
            days = _days_until(parsed)
            if 8 <= days <= 30:
                return UrgencyResult(
                    score=65,
                    level="URGENTE",
                    rationale=f"Data relevante detectada ({raw}) em {days} dia(s), dentro da janela de urgência.",
                )
        closest_days = _days_until(closest[1])
    else:
        closest_days = None

    immediate_keyword = _contains_keyword(body, _KEYWORDS_IMEDIATO)
    if immediate_keyword:
        return UrgencyResult(
            score=85,
            level="IMEDIATO",
            rationale=f"Indicador de urgência imediata encontrado no texto: {immediate_keyword}.",
        )

    urgent_keyword = _contains_keyword(body, _KEYWORDS_URGENTE)
    if urgent_keyword:
        return UrgencyResult(
            score=60,
            level="URGENTE",
            rationale=f"Indicador processual de urgência encontrado no texto: {urgent_keyword}.",
        )

    word_count = len(body.split())
    if word_count < 50:
        return UrgencyResult(
            score=10,
            level="BAIXO",
            rationale="Texto curto ou sem indicadores processuais suficientes para justificar prioridade.",
        )

    detail = f" Tipo contratual informado: {contract_type}." if contract_type else ""
    if closest_days is not None:
        detail += f" Data mais próxima está a {closest_days} dia(s), fora da janela de urgência."
    return UrgencyResult(
        score=30,
        level="NORMAL",
        rationale=f"Não foram encontrados prazos próximos nem palavras-chave fortes de urgência.{detail}",
    )
