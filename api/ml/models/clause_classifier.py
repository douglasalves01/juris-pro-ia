"""Classificacao de clausulas abusivas com fallback deterministico."""

from __future__ import annotations

import re
from typing import Any
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


ClauseLabel = Literal["abusiva", "padrao", "favoravel"]
_CACHE: dict[str, Any] = {}


@dataclass(frozen=True)
class ClauseClassification:
    numero: str
    tipo: str
    label: ClauseLabel
    confidence: float
    rationale: str
    evidence: str


def _has_model_weights(models_dir: str) -> bool:
    base = Path(models_dir).expanduser().resolve()
    direct = base / "classificacao_clausulas"
    return (direct / "config.json").is_file() and (
        (direct / "model.safetensors").is_file() or (direct / "pytorch_model.bin").is_file()
    )


def _model_dir(models_dir: str) -> Path:
    return Path(models_dir).expanduser().resolve() / "classificacao_clausulas"


def _predict_with_model(text: str, models_dir: str) -> tuple[ClauseLabel, float] | None:
    if not _has_model_weights(models_dir):
        return None
    model_path = _model_dir(models_dir)
    key = str(model_path)
    if key not in _CACHE:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        model.eval()
        label_map_path = model_path / "label_map.json"
        if label_map_path.is_file():
            import json

            label_map = json.loads(label_map_path.read_text(encoding="utf-8"))
            id_to_label = {int(v): str(k) for k, v in label_map.items()}
        else:
            id_to_label = {0: "abusiva", 1: "padrao", 2: "favoravel"}
        _CACHE[key] = (tokenizer, model, id_to_label)
    tokenizer, model, id_to_label = _CACHE[key]
    encoded = tokenizer(text, truncation=True, max_length=384, return_tensors="pt")
    with torch.no_grad():
        logits = model(**encoded).logits[0]
        probs = torch.softmax(logits, dim=-1)
        idx = int(torch.argmax(probs).item())
        label = id_to_label.get(idx, "padrao")
        confidence = float(probs[idx].item())
    if label not in {"abusiva", "padrao", "favoravel"}:
        label = "padrao"
    return label, confidence  # type: ignore[return-value]


def _percent_after_multas(text: str) -> int | None:
    match = re.search(r"multa[^.]{0,80}?(\d{1,3})\s*%", text, flags=re.IGNORECASE)
    if not match:
        return None
    return int(match.group(1))


def classify_clause(clause: dict, models_dir: str = "hf_models") -> ClauseClassification:
    text = str(clause.get("texto") or clause.get("text") or "")
    title = str(clause.get("titulo") or "")
    full_text = f"{title} {text}".strip()
    lowered = full_text.lower()
    numero = str(clause.get("numero") or "")
    tipo = str(clause.get("tipo") or "geral")

    model_prediction = _predict_with_model(full_text, models_dir)
    if model_prediction is not None:
        label, confidence = model_prediction
        return ClauseClassification(
            numero=numero,
            tipo=tipo,
            label=label,
            confidence=confidence,
            rationale="Classificacao gerada pelo modelo fine-tuned de clausulas.",
            evidence=full_text[:500],
        )

    pct = _percent_after_multas(full_text)
    if pct is not None and pct > 20:
        return ClauseClassification(
            numero=numero,
            tipo=tipo,
            label="abusiva",
            confidence=0.86 if pct > 30 else 0.78,
            rationale=f"Multa contratual de {pct}% acima do patamar usual de 10-20%.",
            evidence=full_text[:500],
        )

    abusive_patterns = [
        (r"ren[uú]ncia\s+(?:irrevog[aá]vel\s+)?(?:a\s+)?direitos", "Renuncia ampla de direitos."),
        (r"rescis[aã]o\s+unilateral[^.]{0,100}?(?:sem|independente de)\s+(?:aviso|notifica)", "Rescisao unilateral sem aviso adequado."),
        (r"foro\s+exclusivo[^.]{0,100}?(?:exterior|estrangeiro|outro\s+pa[ií]s)", "Foro exclusivo potencialmente oneroso."),
        (r"limita(?:r|cao|ção)[^.]{0,80}?responsabilidade[^.]{0,80}?(?:zero|nenhuma|integralmente exclu)", "Exclusao excessiva de responsabilidade."),
        (r"dados\s+pessoais(?![^.]{0,160}(?:base legal|consentimento|lgpd|art\.?\s*7))", "Tratamento de dados sem base legal aparente."),
    ]
    for pattern, rationale in abusive_patterns:
        if re.search(pattern, lowered, flags=re.IGNORECASE):
            return ClauseClassification(
                numero=numero,
                tipo=tipo,
                label="abusiva",
                confidence=0.74,
                rationale=rationale,
                evidence=full_text[:500],
            )

    favorable_patterns = [
        r"equil[ií]brio contratual",
        r"boa-f[eé]",
        r"media[cç][aã]o",
        r"prazo de aviso[^.]{0,80}?(?:30|60|90)\s+dias",
        r"direitos do titular",
        r"limita[cç][aã]o de responsabilidade[^.]{0,80}?teto",
    ]
    if any(re.search(pattern, lowered, flags=re.IGNORECASE) for pattern in favorable_patterns):
        return ClauseClassification(
            numero=numero,
            tipo=tipo,
            label="favoravel",
            confidence=0.68,
            rationale="Clausula contem mecanismo de equilibrio, transparencia ou mitigacao de risco.",
            evidence=full_text[:500],
        )

    return ClauseClassification(
        numero=numero,
        tipo=tipo,
        label="padrao",
        confidence=0.55,
        rationale="Nenhum indicador forte de abusividade foi detectado por regras locais.",
        evidence=full_text[:500],
    )


def classify_clauses(clauses: list[dict], models_dir: str = "hf_models") -> list[ClauseClassification]:
    return [classify_clause(clause, models_dir=models_dir) for clause in clauses]
