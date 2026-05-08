"""Predição de desfecho processual (ganhou | perdeu | inconclusivo)."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

_CACHE: dict[str, Any] = {}

# Padrões determinísticos — se aparecem no texto, o desfecho é certo
_RULES_PROCEDENTE = [
    r"JULGO\s+PROCEDENTE",
    r"julgo\s+procedente",
    r"julgou\s+procedente",
    r"dou\s+provimento\s+ao\s+recurso",
    r"recurso\s+provido",
    r"pedido\s+(?:é\s+)?procedente",
    r"CONDENO\s+(?:a|o)\s+r[eé][u|]",
    r"condeno\s+(?:a|o)\s+r[eé][u|]",
]

_RULES_IMPROCEDENTE = [
    r"JULGO\s+IMPROCEDENTE",
    r"julgo\s+improcedente",
    r"julgou\s+improcedente",
    r"nego\s+provimento\s+ao\s+recurso",
    r"recurso\s+(?:n[aã]o\s+)?provido",
    r"pedido\s+(?:é\s+)?improcedente",
    r"INDEFIRO\s+o\s+pedido",
]

_RULES_PARCIAL = [
    r"JULGO\s+PARCIALMENTE\s+PROCEDENTE",
    r"julgo\s+parcialmente\s+procedente",
    r"parcialmente\s+procedente",
    r"dou\s+parcial\s+provimento",
    r"provimento\s+parcial",
]


def _rule_based_outcome(text: str) -> dict[str, Any] | None:
    """
    Detecta desfecho diretamente pelo texto do dispositivo.
    Retorna None se não encontrar padrão claro.
    """
    t = text or ""

    # Parcial tem prioridade sobre procedente puro
    for pat in _RULES_PARCIAL:
        if re.search(pat, t, re.I):
            return {
                "win_prediction": "ganhou",
                "win_probability": 0.65,
                "win_confidence": 0.90,
                "win_confidence_source": "regra_textual",
                "outcome_probabilities": {"ganhou": 0.65, "perdeu": 0.10, "inconclusivo": 0.25},
            }

    for pat in _RULES_PROCEDENTE:
        if re.search(pat, t):
            return {
                "win_prediction": "ganhou",
                "win_probability": 0.92,
                "win_confidence": 0.95,
                "win_confidence_source": "regra_textual",
                "outcome_probabilities": {"ganhou": 0.92, "perdeu": 0.04, "inconclusivo": 0.04},
            }

    for pat in _RULES_IMPROCEDENTE:
        if re.search(pat, t):
            return {
                "win_prediction": "perdeu",
                "win_probability": 0.05,
                "win_confidence": 0.93,
                "win_confidence_source": "regra_textual",
                "outcome_probabilities": {"ganhou": 0.05, "perdeu": 0.91, "inconclusivo": 0.04},
            }

    return None


def _get_torch_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _resolve_submodel_path(models_dir: str, name: str) -> Path:
    base = Path(models_dir).expanduser().resolve()
    direct = base / name
    if direct.is_dir():
        return direct
    parent = base.parent
    if (parent / name).is_dir():
        return parent / name
    return direct


def _move_model_to_device(model: torch.nn.Module, device: torch.device) -> torch.nn.Module:
    model = model.to(device)
    if device.type == "cuda":
        model = model.half()
    return model


def _ensure_loaded(models_dir: str) -> tuple[torch.device, Any, Any, dict[int, str]]:
    path = _resolve_submodel_path(models_dir, "predicao_ganho")
    key = str(path)
    if key in _CACHE:
        return _CACHE[key]

    device = _get_torch_device()
    tokenizer = AutoTokenizer.from_pretrained(str(path), use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(str(path))
    model = _move_model_to_device(model, device)
    model.eval()

    id2label = {int(k): v for k, v in model.config.id2label.items()}

    _CACHE[key] = (device, model, tokenizer, id2label)
    return _CACHE[key]


def predict(text: str, models_dir: str) -> dict[str, Any]:
    """Retorna classe prevista e probabilidade de vitória (classe ganhou)."""

    # Regras determinísticas têm prioridade sobre o modelo
    rule_result = _rule_based_outcome(text)
    if rule_result:
        return rule_result

    path = _resolve_submodel_path(models_dir, "predicao_ganho")
    if not path.is_dir():
        return {
            "win_prediction": "inconclusivo",
            "win_probability": 0.33,
            "win_confidence": 0.33,
            "outcome_probabilities": {},
            "error": "predicao_ganho não encontrado",
        }

    device, model, tokenizer, id2label = _ensure_loaded(models_dir)
    max_length = min(getattr(model.config, "max_position_embeddings", 512), 512)

    batch = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding=True,
    )
    batch = {k: v.to(device) for k, v in batch.items()}

    with torch.inference_mode():
        logits = model(**batch).logits
        probs = torch.softmax(logits, dim=-1)[0]
    probs_list = probs.tolist()
    best_idx = int(max(range(len(probs_list)), key=lambda i: probs_list[i]))

    probs_out = {id2label[i]: float(probs_list[i]) for i in range(len(probs_list))}

    ganhou_idx = next((i for i, lab in id2label.items() if lab == "ganhou"), None)
    win_prob = float(probs_list[ganhou_idx]) if ganhou_idx is not None else 0.0
    win_confidence = float(probs_list[best_idx])

    return {
        "win_prediction": id2label[best_idx],
        "win_probability": win_prob,
        "win_confidence": win_confidence,
        "outcome_probabilities": probs_out,
    }
