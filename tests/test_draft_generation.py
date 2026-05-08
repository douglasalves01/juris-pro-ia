from __future__ import annotations

from fastapi.testclient import TestClient

from api.main import app
from api.schemas.analysis import DraftResponse
from api.services.draft_generation_service import build_with_template, generate_draft


def test_build_with_template_marks_missing_fields_for_review() -> None:
    result = build_with_template(
        "peticao_inicial",
        {"parties": "Autor Joao contra Reu Empresa", "facts": "Fatos narrados."},
    )

    assert "[REVISAR]" in result["draft"]
    assert any(section["needsReview"] for section in result["sections"])
    assert "Revisao humana" in result["disclaimer"]


def test_generate_draft_without_api_key_uses_template() -> None:
    result = generate_draft(
        "contrato",
        {
            "parties": ["Contratante A", "Contratada B"],
            "subject": "Prestacao de servicos",
            "facts": "Pagamento mensal.",
            "claims": "Confidencialidade e entrega.",
        },
        "formal",
        None,
        "https://example.invalid/v1",
        "model",
    )

    assert "CONTRATO" in result["draft"]
    assert result["sections"]


def test_generate_draft_endpoint_returns_schema() -> None:
    with TestClient(app) as client:
        response = client.post(
            "/generate/draft",
            json={
                "documentType": "contrato",
                "style": "conciso",
                "context": {
                    "parties": ["Contratante A", "Contratada B"],
                    "subject": "Prestacao de servicos juridicos",
                    "facts": "Honorarios mensais.",
                    "claims": "Entrega de relatorios e confidencialidade.",
                },
            },
        )

    assert response.status_code == 200
    parsed = DraftResponse.model_validate(response.json())
    assert parsed.draft
    assert parsed.sections
    assert parsed.disclaimer


def test_generate_draft_endpoint_rejects_invalid_document_type() -> None:
    with TestClient(app) as client:
        response = client.post(
            "/generate/draft",
            json={"documentType": "invalido", "context": {}},
        )

    assert response.status_code == 422
