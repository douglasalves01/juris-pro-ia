from __future__ import annotations

from fastapi.testclient import TestClient
from hypothesis import given, settings, strategies as st

from api.main import app
from api.ml.models.quality_scorer import score
from api.schemas.analysis import QualityResponse


def test_quality_scorer_pontua_peticao_estruturada_melhor_que_texto_curto() -> None:
    good = score(
        "EXCELENTISSIMO JUIZ DA VARA CIVEL. Autor Joao. Reu Empresa. "
        "DOS FATOS. O autor narra os fatos relevantes. DO DIREITO. Art. 319 do CPC. "
        "DOS PEDIDOS. Requer citacao e condenacao. Valor da causa R$ 1000. "
        "Portanto, os pedidos devem ser acolhidos."
    )
    weak = score("Pedido simples sem fundamento.")

    assert good.score > weak.score
    assert good.dimensions["completeness"] >= 80
    assert weak.suggestions


@given(text=st.text(max_size=2000))
@settings(max_examples=100)
def test_quality_scorer_mantem_scores_no_intervalo(text: str) -> None:
    result = score(text)

    assert 0 <= result.score <= 100
    assert set(result.dimensions) == {"completeness", "coherence", "citations", "language"}
    assert all(0 <= value <= 100 for value in result.dimensions.values())


def test_quality_endpoint_retorna_schema() -> None:
    with TestClient(app) as client:
        response = client.post(
            "/analyze/quality",
            json={"text": "DOS FATOS. Texto com fundamento no art. 319 do CPC. DOS PEDIDOS. Requer condenacao."},
        )

    assert response.status_code == 200
    parsed = QualityResponse.model_validate(response.json())
    assert 0 <= parsed.score <= 100


def test_quality_endpoint_rejeita_texto_vazio() -> None:
    with TestClient(app) as client:
        response = client.post("/analyze/quality", json={"text": " "})

    assert response.status_code == 422
