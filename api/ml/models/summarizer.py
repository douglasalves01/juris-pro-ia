"""Sumarização PT-T5 com prefixo fixo."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

_CACHE: dict[str, Any] = {}


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

def _prefix_for_kind(document_kind: str) -> str:
    if document_kind == "peticao_inicial":
        return "resuma em português claro os fatos e pedidos principais desta petição: "
    if document_kind == "contrato":
        return "resuma em português claro as obrigações e pontos centrais deste contrato: "
    return "resuma o processo jurídico: "


def _petition_summary_suspicious(s: str) -> bool:
    """Heurísticas para resumo de petição com contradições típicas do gerador seq2seq."""
    low = s.casefold()
    if re.search(r"r\$\s*1\.000,00", low) and ("vinte mil" in low or "cento e vinte" in low):
        return True
    if re.search(r"r\$\s*10\.000,00", low) and "dois mil" in low:
        return True
    if "sem pedidos" in low and "requer" in low:
        return True
    return False


def _summary_quality_ok(summary: str, document_kind: str = "outro") -> bool:
    s = (summary or "").strip()
    if len(s) < 50:
        return False
    letters = sum(1 for c in s if c.isalpha())
    if letters < 30:
        return False
    upper = sum(1 for c in s if c.isupper())
    if letters and upper / letters > 0.42:
        return False
    words = s.split()
    if len(words) > 8 and len(set(words)) / len(words) < 0.28:
        return False
    if document_kind == "peticao_inicial" and _petition_summary_suspicious(s):
        return False
    return True


def _fallback_summary(
    focus: str,
    sections: dict[str, list[str]] | None,
    document_kind: str,
) -> str:
    sec = sections or {}
    partes = sec.get("partes", [])[:5]
    valores = sec.get("valores", [])[:5]
    datas = sec.get("datas", [])[:4]
    clausulas = sec.get("clausulas", [])[:3]
    intro = (
        "Síntese automática (peça processual): principais elementos identificados no texto."
        if document_kind == "peticao_inicial"
        else "Síntese automática (contrato ou documento): elementos estruturados extraídos do texto."
    )
    parts: list[str] = [intro]
    if partes:
        parts.append("Partes / identificações: " + "; ".join(partes) + ".")
    if valores:
        parts.append("Valores mencionados: " + "; ".join(valores) + ".")
    if datas:
        parts.append("Datas: " + "; ".join(datas) + ".")
    if clausulas and document_kind == "contrato":
        parts.append("Cláusulas (trechos): " + " | ".join(c[:220] for c in clausulas) + ".")
    tail = " ".join(focus.split()[:100])
    if tail:
        parts.append("Contexto: " + tail + ("…" if len(focus) > len(tail) else "") + ".")
    return " ".join(parts)


def _ensure_loaded(models_dir: str) -> tuple[torch.device, Any, Any, int]:
    path = _resolve_submodel_path(models_dir, "sumarizacao")
    key = str(path)
    if key in _CACHE:
        return _CACHE[key]

    device = _get_torch_device()
    tokenizer = AutoTokenizer.from_pretrained(str(path), use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(str(path))
    model = _move_model_to_device(model, device)
    model.eval()

    npos = int(getattr(model.config, "n_positions", 512))
    max_in = min(1024, npos)

    _CACHE[key] = (device, model, tokenizer, max_in)
    return _CACHE[key]


def predict(
    text: str,
    models_dir: str,
    *,
    document_kind: str = "outro",
    sections: dict[str, list[str]] | None = None,
    use_seq2seq: bool = True,
    deep: bool = False,
) -> dict[str, Any]:
    """Gera resumo executivo a partir do texto focado (encoder truncado).

    use_seq2seq=False: só síntese estruturada (regras), sem T5 — modo rápido.
    deep=True: geração mais longa (modo profundo, ainda 100% local).
    """
    body = (text or "").strip()
    if not use_seq2seq:
        return {"executive_summary": _fallback_summary(body, sections, document_kind)}

    path = _resolve_submodel_path(models_dir, "sumarizacao")
    if not path.is_dir():
        fb = _fallback_summary(body, sections, document_kind)
        return {
            "executive_summary": fb,
            "error": "sumarizacao não encontrado",
        }

    device, model, tokenizer, max_in = _ensure_loaded(models_dir)
    if not body:
        return {"executive_summary": _fallback_summary("", sections, document_kind)}

    prompt = _prefix_for_kind(document_kind) + body
    batch = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_in,
        padding=False,
    )
    batch = {k: v.to(device) for k, v in batch.items()}

    gen_kwargs: dict[str, Any] = {
        "max_new_tokens": 320 if deep else 220,
        "min_length": 32 if deep else 24,
        "do_sample": False,
        "num_beams": 5 if deep else 4,
        "early_stopping": True,
        "repetition_penalty": 1.12,
        "no_repeat_ngram_size": 3,
    }

    with torch.inference_mode():
        out_ids = model.generate(**batch, **gen_kwargs)

    decoded = tokenizer.decode(out_ids[0], skip_special_tokens=True).strip()
    if not _summary_quality_ok(decoded, document_kind):
        decoded = _fallback_summary(body, sections, document_kind)
    return {"executive_summary": decoded}
