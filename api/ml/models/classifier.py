"""Classificação de tipo de contrato/causa (12 classes)."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

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


def _ensure_loaded(models_dir: str) -> tuple[torch.device, Any, Any, dict[int, str]]:
    path = _resolve_submodel_path(models_dir, "classificacao_tipo")
    key = str(path)
    if key in _CACHE:
        return _CACHE[key]

    device = _get_torch_device()
    tokenizer = AutoTokenizer.from_pretrained(str(path), use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(str(path))
    model = _move_model_to_device(model, device)
    model.eval()

    label_map = path / "label_map.json"
    if label_map.is_file():
        with label_map.open(encoding="utf-8") as fh:
            lm = json.load(fh)
        id2label = {int(k): v for k, v in lm["id2label"].items()}
    else:
        id2label = {int(k): v for k, v in model.config.id2label.items()}

    _CACHE[key] = (device, model, tokenizer, id2label)
    return _CACHE[key]


def predict(text: str, models_dir: str) -> dict[str, Any]:
    """Retorna o tipo contratual/causa mais provável e probabilidades."""
    path = _resolve_submodel_path(models_dir, "classificacao_tipo")
    if not path.is_dir():
        return {"contract_type": "Outros", "probs": {}, "error": "classificacao_tipo não encontrado"}

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
        n_out = len(id2label)
        if logits.shape[-1] > n_out:
            logits = logits[..., :n_out]
        probs = torch.softmax(logits, dim=-1)[0]
    probs_list = probs.tolist()
    best_idx = int(max(range(len(probs_list)), key=lambda i: probs_list[i]))

    probs_out = {id2label[i]: float(probs_list[i]) for i in range(len(probs_list))}

    return {
        "contract_type": id2label[best_idx],
        "probs": probs_out,
    }


# Keywords de ALTA PRIORIDADE — se baterem, sobrepõem o modelo
# Ordenadas por especificidade (mais específico primeiro)
_STRONG_KEYWORDS: list[tuple[str, list[str]]] = [
    (
        "Consumidor",
        [
            r"\bC[oó]digo\s+de\s+Defesa\s+do\s+Consumidor\b",
            r"\bCDC\b",
            r"\bnegativa[çc][aã]o\s+indevida\b",
            r"\binscrição\s+(?:indevida\s+)?(?:no|em)\s+(?:SPC|Serasa|cadastro\s+de\s+inadimplentes)\b",
            r"\bSPC\b.*\bSerasa\b",
            r"\bSerasa\b.*\bSPC\b",
            r"\brelação\s+de\s+consumo\b",
            r"\bresponsabilidade\s+(?:civil\s+)?objetiva\b.*\b(?:consumidor|fornecedor)\b",
            r"\bvício\s+do\s+produto\b",
            r"\bvício\s+do\s+servi[çc]o\b",
            r"\bplano\s+de\s+sa[úu]de\b.*\bneg(?:ativa|ou)\b",
            r"\boperadora\s+de\s+telefonia\b",
            r"\btelefonia\b.*\b(?:consumidor|cliente|usu[áa]rio)\b",
        ],
    ),
    (
        "Trabalhista",
        [
            r"\bCLT\b",
            r"\breclamante\b",
            r"\breclamado\b",
            r"\bhoras\s+extras?\b",
            r"\badicional\s+de\s+insalubridade\b",
            r"\brescis[aã]o\s+(?:indireta|do\s+contrato\s+de\s+trabalho)\b",
            r"\bFGTS\b",
            r"\bv[íi]nculo\s+empregat[íi]cio\b",
            r"\bTribunal\s+(?:Regional\s+)?do\s+Trabalho\b",
            r"\bTRT\b",
        ],
    ),
    (
        "Previdenciário",
        [
            r"\bINSS\b",
            r"\baposentadoria\s+por\s+(?:tempo|invalidez|idade)\b",
            r"\bbenef[íi]cio\s+previdenci[áa]rio\b",
            r"\baux[íi]lio[-\s]doen[çc]a\b",
            r"\bBPC\b",
            r"\bprev[íi]d[eê]ncia\s+social\b",
        ],
    ),
    (
        "Tributário",
        [
            r"\bICMS\b",
            r"\bISS\b",
            r"\bIPI\b",
            r"\bReceita\s+Federal\b",
            r"\bexecu[çc][aã]o\s+fiscal\b",
            r"\bCDA\b.*\bd[íi]vida\s+ativa\b",
            r"\bdébito\s+tribut[áa]rio\b",
            r"\bauto\s+de\s+infra[çc][aã]o\s+(?:fiscal|tribut[áa]rio)\b",
        ],
    ),
    (
        "Criminal",
        [
            r"\bC[oó]digo\s+Penal\b",
            r"\bCP\b.*\bart(?:igo)?\.?\s*\d+",
            r"\bréu\b",
            r"\bacusado\b",
            r"\bMinistério\s+P[úu]blico\b.*\bden[úu]ncia\b",
            r"\bcondena[çc][aã]o\s+criminal\b",
            r"\bpena\s+de\s+reclusão\b",
            r"\babsolvição\b",
        ],
    ),
    (
        "Família",
        [
            r"\bdiv[oó]rcio\b",
            r"\bguarda\s+(?:compartilhada|unilateral)\b",
            r"\bpens[aã]o\s+aliment[íi]cia\b",
            r"\balimentos\b.*\bmenor\b",
            r"\bunião\s+est[áa]vel\b",
            r"\binvent[áa]rio\b",
            r"\bsucess[aã]o\s+heredit[áa]ria\b",
        ],
    ),
    (
        "Tecnologia",
        [
            r"\bc[oó]digo[-\s]?fonte\b",
            r"\blicen[çc]a\s+de\s+software\b",
            r"\bSLA\b.*\b(?:sistema|software|plataforma)\b",
            r"\bdesenvolvimento\s+de\s+software\b",
            r"\binfrastrutura\s+de\s+TI\b",
        ],
    ),
    (
        "Serviços",
        [
            r"\bcontrato\s+de\s+presta[çc][aã]o\s+de\s+servi[çc]os\b",
            r"\binadimplemento\s+contratual\b.*\bservi[çc]os\b",
        ],
    ),
    (
        "Parceria",
        [
            r"\bcontrato\s+de\s+parceria\b",
            r"\bacordo\s+de\s+coopera[çc][aã]o\b",
            r"\bjoint\s+venture\b",
        ],
    ),
]


def _strong_keyword_type(text: str) -> str | None:
    """Retorna tipo se keywords de alta prioridade baterem (≥2 matches ou 1 muito específico)."""
    t = text or ""
    for label, pats in _STRONG_KEYWORDS:
        hits = sum(1 for p in pats if re.search(p, t, re.I))
        if hits >= 2 or (hits == 1 and label in ("Criminal", "Previdenciário", "Trabalhista")):
            return label
    return None


def _keyword_suggested_type(text: str) -> str | None:
    """Retorna tipo se ao menos 1 keyword bater (usado como tiebreaker)."""
    t = text or ""
    for label, pats in _STRONG_KEYWORDS:
        for p in pats:
            if re.search(p, t, re.I):
                return label
    return None


def predict_multi_chunk(
    chunks: list[str],
    models_dir: str,
    document_kind: str,
    full_text_sample: str,
) -> dict[str, Any]:
    """
    Média das distribuições de probabilidade em até 5 trechos + voto majoritário,
    com override por keywords fortes e desempate por keywords suaves.
    """
    sample = full_text_sample or "\n\n".join(chunks[:3])

    # Override forte: se keywords de alta prioridade baterem, ignora o modelo
    strong = _strong_keyword_type(sample)
    if strong:
        return {
            "contract_type": strong,
            "probs": {},
            "chunk_votes": [],
            "classification_source": "keyword_override",
        }

    if not chunks:
        return predict(full_text_sample or "", models_dir)

    path = _resolve_submodel_path(models_dir, "classificacao_tipo")
    if not path.is_dir():
        return {"contract_type": "Outros", "probs": {}, "error": "classificacao_tipo não encontrado"}

    n_take = min(5, len(chunks))
    if len(chunks) <= n_take:
        selected = chunks
    else:
        step = max(1, (len(chunks) - 1) // (n_take - 1))
        selected = [chunks[i] for i in range(0, len(chunks), step)][:n_take]

    agg: dict[str, float] = {}
    votes: list[str] = []
    for ch in selected:
        out = predict(ch, models_dir)
        votes.append(str(out.get("contract_type", "Outros")))
        for k, v in (out.get("probs") or {}).items():
            agg[k] = agg.get(k, 0.0) + float(v)

    if agg:
        n = float(len(selected))
        for k in list(agg.keys()):
            agg[k] /= n

    winner = Counter(votes).most_common(1)[0][0]
    vals = sorted(agg.values(), reverse=True) if agg else []
    margin = (vals[0] - vals[1]) if len(vals) > 1 else 1.0

    # Tiebreaker suave: usa keyword se margem pequena
    hint = _keyword_suggested_type(sample)
    if hint and hint != winner and margin < 0.12:
        winner = hint

    return {
        "contract_type": winner,
        "probs": agg,
        "chunk_votes": votes,
        "classification_source": "model",
    }
