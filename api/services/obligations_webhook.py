"""Webhook opcional para enviar obrigacoes extraidas ao backend principal."""

from __future__ import annotations

from typing import Any

import httpx


def notify_obligations(
    webhook_url: str | None,
    *,
    job_id: str | None,
    contract_id: str | None,
    obligations: list[dict[str, Any]],
) -> bool:
    if not webhook_url or not obligations:
        return False
    payload = {
        "jobId": job_id,
        "contractId": contract_id,
        "obligations": obligations,
    }
    try:
        response = httpx.post(webhook_url, json=payload, timeout=5.0)
        response.raise_for_status()
        return True
    except Exception:
        return False
