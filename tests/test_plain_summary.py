from __future__ import annotations

import base64
from unittest.mock import patch

from fastapi.testclient import TestClient
from hypothesis import given, settings, strategies as st

from api.main import app
from api.schemas.analysis import PlainSummaryResponse
from api.services.plain_summary_service import (
    generate_pdf_base64,
    generate_summary,
    simplify_by_rules,
)


_NON_EMPTY_TEXT = st.text(min_size=1, max_size=1000).filter(lambda value: value.strip() != "")
_VALID_LEVELS = st.sampled_from(["leigo", "intermediario", "tecnico"])
_INVALID_LEVELS = st.text(min_size=1, max_size=40).filter(
    lambda value: value not in {"leigo", "intermediario", "tecnico"}
)


@given(text=_NON_EMPTY_TEXT)
@settings(max_examples=100)
def test_fallback_de_regras_sempre_produz_texto_nao_vazio(text: str) -> None:
    result = simplify_by_rules(text, "leigo")

    assert result.strip()


@given(text=_NON_EMPTY_TEXT)
@settings(max_examples=100)
def test_pdf_gerado_e_base64_de_pdf_valido(text: str) -> None:
    result = generate_pdf_base64(text)

    assert base64.b64decode(result).startswith(b"%PDF")


def test_summary_tecnico_retorna_texto_original() -> None:
    text = "O inadimplemento deve ser analisado conforme a contestação."

    assert simplify_by_rules(text, "tecnico") == text


def test_generate_summary_aplica_fallback_sem_api_key() -> None:
    result = generate_summary(
        "O réu apresentou contestação sobre inadimplemento.",
        "leigo",
        None,
        "https://example.invalid/v1",
        "model",
    )

    assert "parte acusada" in result.lower()


def test_generate_pdf_base64_rejeita_texto_vazio() -> None:
    try:
        generate_pdf_base64(" ")
    except ValueError:
        return
    raise AssertionError("generate_pdf_base64 deveria rejeitar texto vazio")


@given(text=_NON_EMPTY_TEXT, level=_VALID_LEVELS, include_pdf=st.booleans())
@settings(max_examples=100)
def test_plain_summary_endpoint_entrada_valida_produz_schema(
    text: str,
    level: str,
    include_pdf: bool,
) -> None:
    with (
        patch(
            "api.main.generate_summary",
            side_effect=lambda executive_summary, level, api_key, base_url, model: (
                f"Resumo {level}: {executive_summary}"
            ),
        ),
        patch(
            "api.main.generate_pdf_base64",
            side_effect=lambda summary_text, brand_name="JurisPro IA": base64.b64encode(b"%PDF fake").decode("ascii"),
        ),
        TestClient(app) as client,
    ):
        response = client.post("/analyze/summary/plain", json={"text": text, "level": level, "include_pdf": include_pdf})

    assert response.status_code == 200
    parsed = PlainSummaryResponse.model_validate(response.json())
    assert parsed.summaryText.strip()
    assert parsed.level == level
    if include_pdf:
        assert parsed.pdfBase64
    else:
        assert parsed.pdfBase64 is None


@given(level=_INVALID_LEVELS)
@settings(max_examples=100)
def test_plain_summary_endpoint_level_invalido_retorna_422(level: str) -> None:
    with TestClient(app) as client:
        response = client.post(
            "/analyze/summary/plain",
            json={"text": "Texto válido para resumo.", "level": level},
        )

    assert response.status_code == 422


def test_plain_summary_endpoint_retorna_schema(monkeypatch) -> None:
    monkeypatch.setattr(
        "api.main.generate_summary",
        lambda executive_summary, level, api_key, base_url, model: f"Resumo {level}: {executive_summary}",
    )

    with TestClient(app) as client:
        response = client.post(
            "/analyze/summary/plain",
            json={"text": "Resumo jurídico técnico.", "level": "leigo", "include_pdf": False},
        )

    assert response.status_code == 200
    parsed = PlainSummaryResponse.model_validate(response.json())
    assert parsed.summaryText.startswith("Resumo leigo:")
    assert parsed.pdfBase64 is None


def test_plain_summary_endpoint_gera_pdf(monkeypatch) -> None:
    monkeypatch.setattr(
        "api.main.generate_summary",
        lambda executive_summary, level, api_key, base_url, model: "Resumo simples.",
    )

    with TestClient(app) as client:
        response = client.post(
            "/analyze/summary/plain",
            json={"text": "Resumo jurídico técnico.", "include_pdf": True},
        )

    assert response.status_code == 200
    parsed = PlainSummaryResponse.model_validate(response.json())
    assert parsed.pdfBase64
    assert base64.b64decode(parsed.pdfBase64).startswith(b"%PDF")


def test_plain_summary_endpoint_sem_entrada_retorna_422() -> None:
    with TestClient(app) as client:
        response = client.post("/analyze/summary/plain", json={})

    assert response.status_code == 422


def test_plain_summary_endpoint_job_inexistente_retorna_404() -> None:
    with TestClient(app) as client:
        response = client.post("/analyze/summary/plain", json={"jobId": "nao-existe"})

    assert response.status_code == 404
