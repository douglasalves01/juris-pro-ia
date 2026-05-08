"""Inferência NER jurídico (BERT token classification) com agregação por chunks."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

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

_MONEY_PATTERN = re.compile(
    r"(?<!\w)(?:R\$\s*)?\d{1,3}(?:\.\d{3})*(?:,\d{2})\b|"
    r"\bR\$\s*\d{1,3}(?:\.\d{3})*(?:,\d{2})\b|"
    r"\bR\$\s*\d+(?:,\d{2})?\b",
    re.IGNORECASE,
)

_GROUP_FOR_LABEL = {
    "PESSOA": "pessoas",
    "ORGANIZACAO": "organizacoes",
    "LEGISLACAO": "legislacao",
    "TEMPO": "datas",
}

_ORG_BLOCK_SUBSTR = (
    "vara ",
    " vara",
    "comarca",
    "tribunal",
    "juiz ",
    "poder jud",
    "forum ",
    "fórum",
    "excelent",
    "ribun",
    "de justica",
    "de justiça",
    "justiça federal",
    "seção jud",
    "secao jud",
    "fls.",
    "folhas",
    "processo n",
    "autos n",
)

_LEG_OK = re.compile(
    r"(lei\s+n[oº°]?\s*)?\s*1{0,1}3\.?709/2018|LGPD|lei\s+8\.?078|CDC|CPC|"
    r"CLT|CF/88|constitui[cç][aã]o|decreto[-\s]?lei|medida\s+provis[oó]ria|\bMP\s*\d+|"
    r"art\.?\s*\d+|c[oó]digo\s+civil|c[oó]digo\s+de\s+defesa",
    re.I,
)


def _filter_org_name(s: str) -> bool:
    low = s.strip().casefold()
    if len(low) < 8:
        return False
    for frag in _ORG_BLOCK_SUBSTR:
        if frag in low:
            return False
    if re.match(r"^(de|da|do|em|no|na|o|a|os|as)\s+\w+$", low):
        return False
    return True


def _filter_leg_item(s: str) -> bool:
    t = s.strip()
    if len(t) < 6:
        return False
    if _LEG_OK.search(t):
        return True
    if len(t) < 14 and not re.search(r"\d", t):
        return False
    return len(t) >= 10


def _normalize_legislacao(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for raw in sorted(items, key=len, reverse=True):
        t = raw.strip()
        if not _filter_leg_item(t):
            continue
        low = t.casefold()
        if "lgpd" in low and "13.709" not in low and "13709" not in low:
            t = "Lei 13.709/2018 (LGPD)"
        key = re.sub(r"\s+", "", t.casefold())
        if key in seen:
            continue
        seen.add(key)
        out.append(t)
    return out[:80]


def _post_filter_entities(entities: dict[str, list[str]]) -> dict[str, list[str]]:
    orgs = [x for x in entities.get("organizacoes", []) if _filter_org_name(x)]
    legs = _normalize_legislacao(list(entities.get("legislacao", [])))
    pessoas = [x for x in entities.get("pessoas", []) if len(x.strip()) >= 4]
    return {
        "pessoas": pessoas[:80],
        "organizacoes": orgs[:80],
        "legislacao": legs,
        "datas": list(entities.get("datas", []))[:80],
        "valores": list(entities.get("valores", []))[:80],
    }


def _ensure_loaded(models_dir: str) -> tuple[torch.device, Any, Any, dict[int, str]]:
    path = _resolve_submodel_path(models_dir, "ner_juridico")
    key = str(path)
    if key in _CACHE:
        return _CACHE[key]

    device = _get_torch_device()
    tokenizer = AutoTokenizer.from_pretrained(str(path), use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(str(path))
    model = _move_model_to_device(model, device)
    model.eval()

    id2label = {int(k): v for k, v in model.config.id2label.items()}

    _CACHE[key] = (device, model, tokenizer, id2label)
    return _CACHE[key]


def _label_to_group(label: str) -> str | None:
    if label == "O":
        return None
    if label.startswith("B-") or label.startswith("I-"):
        return label.split("-", 1)[1]
    return None


def _run_chunk(
    text: str,
    device: torch.device,
    model: torch.nn.Module,
    tokenizer: Any,
    id2label: dict[int, str],
    max_length: int,
) -> list[tuple[str, str]]:
    if not text.strip():
        return []

    batch = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding=False,
        return_offsets_mapping=True,
    )
    offset_mapping = batch.pop("offset_mapping")[0]
    batch = {k: v.to(device) for k, v in batch.items()}

    with torch.inference_mode():
        logits = model(**batch).logits
        preds = logits.argmax(-1)[0].tolist()

    input_ids = batch["input_ids"][0].tolist()
    special_mask = tokenizer.get_special_tokens_mask(input_ids, already_has_special_tokens=True)

    spans: list[tuple[str, str]] = []
    i = 0
    n = len(preds)
    while i < n:
        if special_mask[i]:
            i += 1
            continue
        lab = id2label.get(preds[i], "O")
        grp = _label_to_group(lab)
        if grp is None:
            i += 1
            continue
        start_char = int(offset_mapping[i][0])
        end_char = int(offset_mapping[i][1])
        j = i + 1
        while j < n and not special_mask[j]:
            lab_j = id2label.get(preds[j], "O")
            grp_j = _label_to_group(lab_j)
            if grp_j != grp:
                break
            end_char = int(offset_mapping[j][1])
            j += 1
        span_text = text[start_char:end_char].strip()
        if span_text:
            spans.append((span_text, grp))
        i = j

    return spans


def predict(text: str, models_dir: str) -> dict[str, Any]:
    """
    Extrai entidades agregando chunks de até 512 tokens.

    Retorna chaves alinhadas ao schema: pessoas, organizacoes, legislacao, datas, valores.
    """
    path = _resolve_submodel_path(models_dir, "ner_juridico")
    if not path.is_dir():
        return {
            "entities": {
                "pessoas": [],
                "organizacoes": [],
                "legislacao": [],
                "datas": [],
                "valores": [],
            },
            "error": "ner_juridico não encontrado",
        }

    device, model, tokenizer, id2label = _ensure_loaded(models_dir)
    max_length = min(getattr(model.config, "max_position_embeddings", 512), 512)

    words = text.split()
    stride = 400
    chunk_size = max_length - 32
    chunks: list[str] = []
    if len(words) <= chunk_size:
        chunks.append(" ".join(words))
    else:
        start = 0
        while start < len(words):
            end = min(start + chunk_size, len(words))
            chunks.append(" ".join(words[start:end]))
            if end >= len(words):
                break
            start += stride

    bucket: dict[str, set[str]] = {
        "pessoas": set(),
        "organizacoes": set(),
        "legislacao": set(),
        "datas": set(),
        "valores": set(),
    }

    for ch in chunks:
        for span, grp in _run_chunk(ch, device, model, tokenizer, id2label, max_length):
            key = _GROUP_FOR_LABEL.get(grp)
            if key:
                bucket[key].add(span)

    for m in _MONEY_PATTERN.finditer(text):
        bucket["valores"].add(m.group(0).strip())

    raw_entities = {
        "pessoas": sorted(bucket["pessoas"], key=len, reverse=True)[:80],
        "organizacoes": sorted(bucket["organizacoes"], key=len, reverse=True)[:80],
        "legislacao": sorted(bucket["legislacao"], key=len, reverse=True)[:80],
        "datas": sorted(bucket["datas"], key=len, reverse=True)[:80],
        "valores": sorted(bucket["valores"], key=len, reverse=True)[:80],
    }
    return {"entities": _post_filter_entities(raw_entities)}
