#!/usr/bin/env python3
"""
Teste end-to-end do JurisPro IA (HTTP real, sem mocks).

Pré-requisitos: Postgres, Redis, API, worker; migrations e seed opcional para similar_cases.

  cp .env.example .env
  docker compose up -d --build
  # após API subir (create_all) e com tabelas base:
  alembic upgrade head
  python scripts/seed_cases.py
  python scripts/test_integration.py
"""

from __future__ import annotations

import io
import os
import sys
import time
import uuid
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

BASE_URL = os.environ.get("BASE_URL", "http://127.0.0.1:8000").rstrip("/")
TIMEOUT_SEC = int(os.environ.get("E2E_TIMEOUT_SEC", "180"))
POLL_INTERVAL = float(os.environ.get("E2E_POLL_INTERVAL", "2"))

EXPECTED_RESULT_KEYS = (
    "contract_type",
    "risk_score",
    "risk_level",
    "attention_points",
    "executive_summary",
    "win_probability",
    "fee_estimate_min",
    "fee_estimate_max",
    "similar_cases",
    "entities",
)


def _make_pdf_bytes() -> bytes:
    from fpdf import FPDF

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=11)
    text = (
        "CONTRATO DE PRESTACAO DE SERVICOS DE DESENVOLVIMENTO DE SOFTWARE\n\n"
        "Clausula 1. O prestador desenvolvera sistema de gestao conforme especificacao. "
        "Multa de R$ 10.000,00 por atraso na entrega.\n"
        "Clausula 2. Vigencia de 24 meses com renovacao automatica.\n"
        "Parte contratante: Empresa Alfa Ltda. Parte contratada: Tech Solucoes S.A.\n"
        "Foro da comarca de Sao Paulo, em caso de litigio e execucao fiscal eventual."
    )
    pdf.multi_cell(0, 6, text)
    out = pdf.output(dest="S")
    if isinstance(out, (bytes, bytearray)):
        return bytes(out)
    return out.encode("latin-1")


def main() -> int:
    email = f"e2e_{uuid.uuid4().hex[:12]}@example.com"
    password = "SenhaSegura123!"

    print(f"BASE_URL={BASE_URL}")
    print(f"Usuário de teste: {email}")

    with httpx.Client(base_url=BASE_URL, timeout=120.0) as client:
        r = client.post(
            "/auth/register",
            json={
                "firm_name": "Escritório E2E",
                "cnpj": "00.000.000/0001-91",
                "region": "SP",
                "email": email,
                "password": password,
                "full_name": "Admin E2E",
            },
        )
        print(f"POST /auth/register → {r.status_code}")
        r.raise_for_status()
        access = r.json().get("access_token")

        if not access:
            r = client.post("/auth/login", json={"email": email, "password": password})
            print(f"POST /auth/login → {r.status_code}")
            r.raise_for_status()
            access = r.json()["access_token"]

        headers = {"Authorization": f"Bearer {access}"}

        pdf_bytes = _make_pdf_bytes()
        files = {"file": ("contrato_e2e.pdf", io.BytesIO(pdf_bytes), "application/pdf")}
        r = client.post("/documents/upload", headers=headers, files=files)
        print(f"POST /documents/upload → {r.status_code}")
        r.raise_for_status()
        doc_id = str(r.json()["document_id"])

        r = client.post(f"/analysis/start/{doc_id}", headers=headers)
        print(f"POST /analysis/start/{doc_id} → {r.status_code}")
        r.raise_for_status()
        start = r.json()
        task_id = start["task_id"]
        analysis_id = str(start["analysis_id"])

        deadline = time.monotonic() + TIMEOUT_SEC
        status_payload = None
        while time.monotonic() < deadline:
            r = client.get(f"/analysis/status/{task_id}", headers=headers)
            r.raise_for_status()
            status_payload = r.json()
            st = status_payload.get("status")
            print(f"  status={st}")
            if st == "done":
                break
            if st == "error":
                print("Falha na análise:", status_payload)
                return 1
            time.sleep(POLL_INTERVAL)
        else:
            print("Timeout aguardando análise.")
            return 1

        r = client.get(f"/analysis/{analysis_id}", headers=headers)
        print(f"GET /analysis/{analysis_id} → {r.status_code}")
        r.raise_for_status()
        body = r.json()
        result = body.get("result") or {}

    print("\n--- Resultado (result) ---")
    for k in sorted(result.keys()):
        v = result[k]
        preview = str(v)[:200] + ("..." if len(str(v)) > 200 else "")
        print(f"  {k}: {preview}")

    print("\n--- Validação por campo ---")
    all_ok = True
    for key in EXPECTED_RESULT_KEYS:
        present = key in result and result[key] is not None
        if key == "attention_points" and isinstance(result.get(key), list):
            present = True
        if key == "similar_cases" and isinstance(result.get(key), list):
            present = True
        if key == "entities" and isinstance(result.get(key), dict):
            present = True
        if key == "win_probability" and isinstance(result.get(key), (int, float)):
            present = True
        status = "PASS" if present else "FAIL"
        if not present:
            all_ok = False
        print(f"  {key}: {status}")

    if all_ok:
        print("\nRESULTADO GLOBAL: PASS")
        return 0
    print("\nRESULTADO GLOBAL: FAIL")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
