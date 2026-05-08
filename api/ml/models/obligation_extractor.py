"""Extracao heuristica de obrigacoes, sujeitos e prazos."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, timedelta

from api.ml.models.urgency_classifier import _parse_date


@dataclass(frozen=True)
class Obligation:
    subject: str
    obligation: str
    deadline: str | None = None
    deadlineAbsolute: str | None = None
    confidence: float = 0.5


_SUBJECT_PATTERN = r"(contratante|contratada|contratado|locador|locat[aá]rio|empregado|empregador|fornecedor|cliente|parte)"
_VERB_PATTERN = r"(dever[aá]|deve|obriga-se|compromete-se|responsabiliza-se|fica obrigada|fica obrigado)"


def _reference_date(dates: list[str]) -> date | None:
    parsed = [_parse_date(item) for item in dates or []]
    parsed = [item for item in parsed if item is not None]
    return min(parsed) if parsed else None


def _absolute_deadline(deadline_text: str | None, reference: date | None) -> str | None:
    if not deadline_text:
        return None
    explicit = _parse_date(deadline_text)
    if explicit is not None:
        return explicit.isoformat()
    if reference is None:
        return None
    match = re.search(r"(\d{1,3})\s+dias?", deadline_text, flags=re.IGNORECASE)
    if not match:
        return None
    return (reference + timedelta(days=int(match.group(1)))).isoformat()


def extract(text: str, entities_datas: list[str] | None = None) -> list[Obligation]:
    body = text or ""
    reference = _reference_date(entities_datas or [])
    obligations: list[Obligation] = []
    pattern = re.compile(
        rf"\b(?P<subject>{_SUBJECT_PATTERN})\b[^.\n]{{0,80}}?\b(?P<verb>{_VERB_PATTERN})\b"
        rf"(?P<obligation>[^.\n]{{8,220}}?)(?:\s+(?:em|no prazo de|ate|até)\s+(?P<deadline>"
        rf"\d{{1,3}}\s+dias?(?:\s+ap[oó]s\s+a\s+assinatura)?|\d{{1,2}}[/-]\d{{1,2}}[/-]\d{{2,4}}))?[.;]",
        flags=re.IGNORECASE,
    )
    for match in pattern.finditer(body):
        subject = re.sub(r"\s+", " ", match.group("subject")).strip()
        obligation_text = re.sub(r"\s+", " ", match.group("obligation")).strip(" ,;:")
        deadline = match.group("deadline")
        if not obligation_text:
            continue
        obligations.append(
            Obligation(
                subject=subject,
                obligation=obligation_text[:300],
                deadline=deadline.strip() if deadline else None,
                deadlineAbsolute=_absolute_deadline(deadline, reference),
                confidence=0.75 if deadline else 0.62,
            )
        )
    return obligations[:30]
