"""Análise de risco contratual: BERT + regras de supervisão fraca."""

from __future__ import annotations

import re
from collections.abc import Callable
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

_LEVEL_ORDER = ("baixo", "médio", "alto", "crítico")
_LEVEL_TO_SCORE = {"baixo": 18, "médio": 45, "alto": 72, "crítico": 92}


def detectar_lgpd(texto: str) -> dict[str, Any] | None:
    tem_dados_pessoais = bool(
        re.search(
            r"dados pessoais|tratamento de dados|coleta de dados|dados sens[íi]veis",
            texto,
            re.I,
        )
    )
    tem_base_legal = bool(
        re.search(
            r"base legal|consentimento|leg[íi]timo interesse|contrato|obrigação legal|"
            r"DPO|encarregado de dados|Art\. 7|Art\. 11|LGPD",
            texto,
            re.I,
        )
    )
    if tem_dados_pessoais and not tem_base_legal:
        return {
            "tipo": "lgpd_compliance",
            "severidade": "alta",
            "descricao": (
                "Possível não conformidade com LGPD detectada: tratamento de dados "
                "pessoais sem base legal explícita."
            ),
            "_needle": r"(?:LGPD|dados pessoais|tratamento de dados)",
        }
    return None


def detectar_multa_abusiva(texto: str) -> dict[str, Any] | None:
    matches = re.findall(r"multa[^.]{0,60}?(\d{1,3})\s*%", texto, re.I)
    for m in matches:
        pct = int(m)
        if pct > 20:
            return {
                "tipo": "penalty_clause",
                "severidade": "alta" if pct > 30 else "média",
                "descricao": (
                    f"Cláusula de multa com {pct}% — acima do padrão de mercado (10-20%). "
                    f"Risco de nulidade parcial."
                ),
                "_needle": r"multa[^.]{0,80}?" + re.escape(str(pct)) + r"\s*%",
            }
    return None


def detectar_rescisao_unilateral(texto: str) -> dict[str, Any] | None:
    matches = re.findall(
        r"rescis[aã]o\s+unilateral[^.]{0,100}?(\d{1,3})\s*dias?",
        texto,
        re.I,
    )
    for m in matches:
        dias = int(m)
        if dias < 30:
            return {
                "tipo": "termination_clause",
                "severidade": "alta" if dias < 15 else "média",
                "descricao": (
                    f"Rescisão unilateral com apenas {dias} dias de aviso — abaixo do "
                    f"prazo razoável de 30 dias."
                ),
                "_needle": r"rescis[aã]o\s+unilateral",
            }
    return None


def detectar_foro_desfavoravel(texto: str) -> dict[str, Any] | None:
    if re.search(r"foro.*?exterior|jurisdição.*?estranger|lei estrangeira", texto, re.I):
        return {
            "tipo": "jurisdiction_clause",
            "severidade": "alta",
            "descricao": (
                "Cláusula de eleição de foro estrangeiro ou aplicação de lei estrangeira — "
                "pode dificultar a defesa."
            ),
            "_needle": r"foro|jurisdição|lei estrangeira",
        }
    return None


def detectar_ausencia_limite_responsabilidade(texto: str) -> dict[str, Any] | None:
    tem_contrato_valor_alto = bool(re.search(r"R\$\s*[\d.,]{6,}", texto))
    tem_limite = bool(
        re.search(
            r"limita[çc][aã]o de responsabilidade|responsabilidade m[aá]xima|"
            r"teto de responsabilidade|cap de responsabilidade",
            texto,
            re.I,
        )
    )
    if tem_contrato_valor_alto and not tem_limite:
        return {
            "tipo": "liability_limitation",
            "severidade": "média",
            "descricao": (
                "Ausência de cláusula de limitação de responsabilidade em contrato de alto valor."
            ),
            "_needle": r"R\$\s*[\d.,]{6,}",
        }
    return None


def detectar_propriedade_intelectual(texto: str) -> dict[str, Any] | None:
    tem_pi = bool(
        re.search(
            r"propriedade intelectual|software|código-fonte|invenção|criação|"
            r"direitos autorais|patente",
            texto,
            re.I,
        )
    )
    tem_cessao_clara = bool(
        re.search(
            r"cede|transfere|pertence.*?contratante|titularidade.*?contratante",
            texto,
            re.I,
        )
    )
    if tem_pi and not tem_cessao_clara:
        return {
            "tipo": "intellectual_property",
            "severidade": "média",
            "descricao": (
                "Titularidade de propriedade intelectual não definida claramente — "
                "risco de conflito futuro."
            ),
            "_needle": r"propriedade intelectual|software|código-fonte",
        }
    return None


def detectar_arbitragem(texto: str) -> dict[str, Any] | None:
    tem_valor_alto = bool(re.search(r"R\$\s*[\d.,]{7,}", texto))
    tem_arbitragem = bool(
        re.search(
            r"arbitragem|árbitro|câmara arbitral|CAMARB|CIESP|ICC|CCI",
            texto,
            re.I,
        )
    )
    if tem_valor_alto and not tem_arbitragem:
        return {
            "tipo": "dispute_resolution",
            "severidade": "baixa",
            "descricao": (
                "Ausência de cláusula arbitral em contrato de alto valor — considere incluir "
                "para agilizar resolução de conflitos."
            ),
            "_needle": r"R\$\s*[\d.,]{7,}",
        }
    return None


# (função, tipos de documento em que a regra NÃO se aplica — ex.: cláusulas em petição)
_REGRAS_COM_ESCOPO: list[tuple[Callable[[str], dict[str, Any] | None], frozenset[str]]] = [
    (detectar_lgpd, frozenset()),
    (detectar_multa_abusiva, frozenset()),
    (detectar_rescisao_unilateral, frozenset({"peticao_inicial"})),
    (detectar_foro_desfavoravel, frozenset()),
    (detectar_ausencia_limite_responsabilidade, frozenset({"peticao_inicial"})),
    (detectar_propriedade_intelectual, frozenset({"peticao_inicial"})),
    (detectar_arbitragem, frozenset({"peticao_inicial"})),
]


def _snippet_for_needle(texto: str, needle_pattern: str) -> str | None:
    m = re.search(needle_pattern, texto, re.I)
    if not m:
        return None
    a = max(0, m.start() - 120)
    b = min(len(texto), m.end() + 180)
    snippet = texto[a:b].strip()
    if len(snippet) > 420:
        snippet = snippet[:417] + "…"
    return snippet


def _rule_score_from_severidade(sev: str) -> int:
    if sev in ("crítica", "crítico", "alta"):
        return 28
    if sev == "média":
        return 15
    if sev == "baixa":
        return 6
    return 10


def _aggregate_rule_score(achados: list[dict[str, Any]]) -> int:
    score = 0
    for a in achados:
        score += _rule_score_from_severidade(str(a.get("severidade", "média")))
    return min(score, 100)


def _ensure_loaded(models_dir: str) -> tuple[torch.device, Any, Any, dict[int, str]]:
    path = _resolve_submodel_path(models_dir, "analise_risco")
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


def predict(
    text: str,
    models_dir: str,
    classification_text: str | None = None,
    document_kind: str = "outro",
) -> dict[str, Any]:
    """
    Combina classificador de risco com regras heurísticas.

    `text`: documento completo para as regras de supervisão fraca.
    `classification_text`: trecho representativo para o BERT (como os demais classificadores).

    Retorna risk_level, risk_score (0-100), attention_points e probabilidades do modelo.
    """
    texto = text or ""
    modelo_in = (classification_text if classification_text is not None else texto) or ""
    achados_raw: list[dict[str, Any]] = []
    ref_tipo = "clausula" if document_kind == "contrato" else "trecho_processual"
    for regra, skip_kinds in _REGRAS_COM_ESCOPO:
        if document_kind in skip_kinds:
            continue
        r = regra(texto)
        if r:
            needle = r.pop("_needle", None)
            clausula = None
            if isinstance(needle, str):
                clausula = _snippet_for_needle(texto, needle)
            achados_raw.append(
                {
                    "tipo": r["tipo"],
                    "severidade": r["severidade"],
                    "descricao": r["descricao"],
                    "clausula_referencia": clausula,
                    "referencia_tipo": ref_tipo,
                }
            )

    path = _resolve_submodel_path(models_dir, "analise_risco")
    model_probs: dict[str, float] = {}
    risk_level_model = "médio"
    model_score = 50.0

    if path.is_dir():
        device, model, tokenizer, id2label = _ensure_loaded(models_dir)
        max_length = min(getattr(model.config, "max_position_embeddings", 512), 512)
        batch = tokenizer(
            modelo_in,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=True,
        )
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.inference_mode():
            logits = model(**batch).logits
            probs = torch.softmax(logits, dim=-1)[0]
        plist = probs.tolist()
        model_probs = {id2label[i]: float(plist[i]) for i in range(len(plist))}
        best_idx = int(max(range(len(plist)), key=lambda i: plist[i]))
        risk_level_model = id2label[best_idx]
        model_score = sum(
            _LEVEL_TO_SCORE.get(id2label[i], 40) * float(plist[i]) for i in range(len(plist))
        )

    rule_score = _aggregate_rule_score(achados_raw)
    risk_score = int(round(0.55 * model_score + 0.45 * float(rule_score)))
    risk_score = max(0, min(100, risk_score))

    if risk_level_model not in _LEVEL_ORDER:
        risk_level_model = _LEVEL_ORDER[min(len(_LEVEL_ORDER) - 1, risk_score // 25)]

    if (
        path.is_dir()
        and model_probs
        and risk_level_model in ("alto", "crítico")
        and max(model_probs.values()) >= 0.42
        and not any(p.get("tipo") == "model_risk_signal" for p in achados_raw)
    ):
        achados_raw.append(
            {
                "tipo": "model_risk_signal",
                "severidade": "média",
                "descricao": (
                    f"O classificador neural indica risco {risk_level_model} "
                    f"(confiança {max(model_probs.values()):.0%})."
                ),
                "clausula_referencia": None,
                "referencia_tipo": ref_tipo,
            }
        )

    return {
        "risk_level": risk_level_model,
        "risk_score": risk_score,
        "attention_points": achados_raw,
        "model_probs": model_probs,
    }
