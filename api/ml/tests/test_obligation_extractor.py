from __future__ import annotations

from fastapi.testclient import TestClient

from api.main import app
from api.ml.models.obligation_extractor import extract
from api.schemas.analysis import ObligationsResponse


def test_extract_obligation_with_relative_deadline() -> None:
    result = extract(
        "Contrato assinado em 01/05/2026. A contratada deve entregar o relatório final no prazo de 30 dias após a assinatura.",
        ["01/05/2026"],
    )

    assert result
    assert result[0].subject.lower() == "contratada"
    assert "entregar" in result[0].obligation
    assert result[0].deadlineAbsolute == "2026-05-31"


def test_extract_obligation_without_deadline() -> None:
    result = extract("O fornecedor obriga-se a manter confidencialidade das informacoes.")

    assert result
    assert result[0].deadline is None


def test_obligations_endpoint_returns_schema() -> None:
    with TestClient(app) as client:
        response = client.post(
            "/analyze/obligations",
            json={"text": "O cliente deve pagar a mensalidade em 10 dias."},
        )

    assert response.status_code == 200
    parsed = ObligationsResponse.model_validate(response.json())
    assert parsed.obligations


def test_obligations_endpoint_rejects_empty_text() -> None:
    with TestClient(app) as client:
        response = client.post("/analyze/obligations", json={"text": " "})

    assert response.status_code == 422
