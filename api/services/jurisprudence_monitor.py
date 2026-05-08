"""Monitoramento simples de novas decisoes para casos inscritos."""

from __future__ import annotations

import hashlib
import re
from typing import Any

import httpx


def _tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-zA-ZÀ-ÿ0-9]{4,}", (text or "").lower())
        if token not in {"para", "como", "pela", "pelo", "sobre", "entre", "este", "esta"}
    }


def lexical_similarity(left: str, right: str) -> float:
    a = _tokens(left)
    b = _tokens(right)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def normalize_decision(raw: dict[str, Any]) -> dict[str, Any]:
    summary = str(raw.get("summary") or raw.get("resumo") or raw.get("ementa") or raw.get("text") or "")
    decision_id = str(raw.get("id") or raw.get("decisionId") or hashlib.sha256(summary.encode("utf-8")).hexdigest()[:16])
    return {
        "id": decision_id,
        "tribunal": str(raw.get("tribunal") or ""),
        "summary": summary,
        "url": raw.get("url"),
    }


def build_alerts(
    subscriptions: dict[str, dict[str, Any]],
    decisions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    alerts: list[dict[str, Any]] = []
    for case_id, subscription in subscriptions.items():
        reference = str(subscription.get("referenceText") or subscription.get("contractId") or case_id)
        threshold = float(subscription.get("threshold") or 0.85)
        for raw_decision in decisions:
            decision = normalize_decision(raw_decision)
            similarity = lexical_similarity(reference, decision["summary"])
            if similarity >= threshold:
                alerts.append(
                    {
                        "caseId": case_id,
                        "newDecisionId": decision["id"],
                        "similarity": round(similarity, 4),
                        "tribunal": decision["tribunal"],
                        "summary": decision["summary"][:1000],
                        "url": decision["url"],
                    }
                )
    return alerts


def notify_alert(webhook_url: str | None, alert: dict[str, Any]) -> bool:
    if not webhook_url:
        return False
    try:
        response = httpx.post(webhook_url, json=alert, timeout=5.0)
        response.raise_for_status()
        return True
    except Exception:
        return False
