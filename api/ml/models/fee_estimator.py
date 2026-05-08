"""Estimativa de honorários (GradientBoosting + encoders) com fallback OAB."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import joblib
import numpy as np

_CACHE: dict[str, Any] = {}

_MONEY_PATTERN = re.compile(
    r"R\$\s*([\d]{1,3}(?:\.\d{3})*(?:,\d{2})?|\d+(?:,\d{2})?)",
    re.IGNORECASE,
)


def _honorarios_dir(models_dir: str) -> Path:
    base = Path(models_dir).resolve()
    cand = base / "honorarios"
    if cand.is_dir():
        return cand
    parent_h = base.parent / "honorarios"
    if parent_h.is_dir():
        return parent_h
    return cand


def _parse_max_brl_value(text: str) -> float:
    best = 50_000.0
    for m in _MONEY_PATTERN.finditer(text or ""):
        raw = m.group(1)
        norm = raw.replace(".", "").replace(",", ".")
        try:
            v = float(norm)
            if not math.isnan(v) and v > best:
                best = v
        except ValueError:
            continue
    return best


def _infer_complexidade(risk_score: int, text_len: int) -> str:
    if risk_score >= 75 or text_len > 25_000:
        return "muito_alta"
    if risk_score >= 50 or text_len > 12_000:
        return "alta"
    if risk_score >= 30 or text_len > 5_000:
        return "média"
    return "simples"


def _le_fit(le: Any, value: str, default: str) -> int:
    classes = list(le.classes_)
    if value in classes:
        return int(le.transform([value])[0])
    if default in classes:
        return int(le.transform([default])[0])
    return 0


def _ensure_loaded(models_dir: str) -> dict[str, Any] | None:
    hdir = _honorarios_dir(models_dir)
    key = str(hdir)
    if key in _CACHE:
        return _CACHE[key]

    min_p = hdir / "model_fee_min.joblib"
    max_p = hdir / "model_fee_max.joblib"
    if not min_p.is_file() or not max_p.is_file():
        _CACHE[key] = None
        return None

    bundle = {
        "model_min": joblib.load(min_p),
        "model_max": joblib.load(max_p),
        "le_tipo": joblib.load(hdir / "le_tipo.joblib"),
        "le_regiao": joblib.load(hdir / "le_regiao.joblib"),
        "le_complexidade": joblib.load(hdir / "le_complexidade.joblib"),
        "oab_tables": {},
    }
    oab_path = hdir / "oab_tables.json"
    if oab_path.is_file():
        with oab_path.open(encoding="utf-8") as fh:
            bundle["oab_tables"] = json.load(fh)

    _CACHE[key] = bundle
    return bundle


def _fallback_from_oab(
    oab_tables: dict[str, Any],
    contract_type: str,
    regiao: str,
    valor_causa: float,
) -> tuple[float, float, float]:
    key = f"{contract_type}|{regiao}"
    row = oab_tables.get(key)
    if not row:
        row = oab_tables.get(f"Outros|{regiao}") or {"fee_min": 2500.0, "fee_max": 15000.0, "base": "fixo"}
    fee_min = float(row.get("fee_min", 2500))
    fee_max = float(row.get("fee_max", 15000))
    base = row.get("base", "fixo")
    if base == "causa" and valor_causa > 0:
        fee_min = max(fee_min, valor_causa * 0.06)
        fee_max = max(fee_max, valor_causa * 0.14)
    suggested = (fee_min + fee_max) / 2.0
    return fee_min, fee_max, suggested


def predict(
    text: str,
    models_dir: str,
    contract_type: str,
    regiao: str,
    risk_score: int,
) -> dict[str, Any]:
    """Prevê faixa de honorários a partir do tipo de causa, região e complexidade inferida."""
    bundle = _ensure_loaded(models_dir)
    valor_causa = _parse_max_brl_value(text)
    comp = _infer_complexidade(risk_score, len(text or ""))
    regiao_n = (regiao or "SP").strip().upper()
    if len(regiao_n) > 2:
        regiao_n = regiao_n[:2]

    tipo = contract_type or "Outros"

    if bundle is None:
        oab_path = _honorarios_dir(models_dir) / "oab_tables.json"
        oab: dict[str, Any] = {}
        if oab_path.is_file():
            with oab_path.open(encoding="utf-8") as fh:
                oab = json.load(fh)
        fmin, fmax, sugg = _fallback_from_oab(oab, tipo, regiao_n, valor_causa)
        return {
            "fee_estimate_min": round(fmin, 2),
            "fee_estimate_max": round(fmax, 2),
            "fee_estimate_suggested": round(sugg, 2),
        }

    le_t = bundle["le_tipo"]
    le_r = bundle["le_regiao"]
    le_c = bundle["le_complexidade"]

    ti = _le_fit(le_t, tipo, "Outros")
    ri = _le_fit(le_r, regiao_n, "SP")
    ci = _le_fit(le_c, comp, "média")

    x = np.column_stack([[ti], [ri], [ci], [np.log1p(valor_causa)]])
    pred_min = float(bundle["model_min"].predict(x)[0])
    pred_max = float(bundle["model_max"].predict(x)[0])
    if pred_max < pred_min:
        pred_min, pred_max = pred_max, pred_min

    suggested = (pred_min + pred_max) / 2.0

    return {
        "fee_estimate_min": round(pred_min, 2),
        "fee_estimate_max": round(pred_max, 2),
        "fee_estimate_suggested": round(suggested, 2),
    }
