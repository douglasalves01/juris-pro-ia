"""Testes HTTP do contrato de análise: saúde, sucesso mockado, erros e trace."""

from __future__ import annotations

import io
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from api.main import app
from api.ml.pipeline import AnalysisPipeline, AnalysisResult, EntitiesBlock

AnalysisPipeline._instance = None


def _minimal_analysis_result() -> AnalysisResult:
    return AnalysisResult(
        contract_type="Consumidor",
        document_kind="outro",
        risk_score=10,
        risk_level="baixo",
        attention_points=[],
        executive_summary="Resumo.",
        main_risks=[],
        recommendations=[],
        positive_points=[],
        win_prediction="inconclusivo",
        win_probability=0.33,
        win_confidence=0.4,
        outcome_probabilities={},
        fee_estimate_min=1.0,
        fee_estimate_max=2.0,
        fee_estimate_suggested=1.5,
        entities=EntitiesBlock(),
        similar_cases=[],
        processing_time_seconds=0.1,
    )


@pytest.fixture
def client():
    with TestClient(app) as tc:
        mock_p = MagicMock()
        mock_p.last_steps = []
        mock_p.analyze.return_value = _minimal_analysis_result()
        tc.app.state.pipeline = mock_p
        yield tc


def test_health_ok() -> None:
    with TestClient(app) as client:
        response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["models_loaded"] is True


def test_analyze_file_txt_returns_done_and_non_empty_trace(client: TestClient) -> None:
    files = {
        "file": (
            "doc.txt",
            io.BytesIO(b"Peticao inicial. Autor pede danos morais e materiais."),
            "text/plain",
        )
    }
    response = client.post("/analyze/file", files=files, data={"regiao": "SP"})
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "done"
    assert body["jobId"]
    assert body["trace"]["steps"]
    assert body["result"]["document"]["legalArea"] == "Consumidor"


def test_analyze_file_validates_response_schema(client: TestClient) -> None:
    from api.schemas.analysis import AnalysisResponse

    files = {
        "file": (
            "doc.txt",
            io.BytesIO(b"Conteudo suficiente para analise do contrato de prestacao."),
            "text/plain",
        )
    }
    response = client.post("/analyze/file", files=files, data={"regiao": "SP"})
    assert response.status_code == 200
    AnalysisResponse.model_validate(response.json())


def test_analyze_file_stream_emits_steps_and_done(client: TestClient) -> None:
    files = {
        "file": (
            "doc.txt",
            io.BytesIO(b"Conteudo suficiente para analisar via SSE."),
            "text/plain",
        )
    }
    response = client.post("/analyze/file/stream", files=files, data={"regiao": "SP"})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    body = response.text
    assert "event: queued" in body
    assert "event: processing" in body
    assert "event: step" in body
    assert "event: done" in body


def test_analyze_file_unsupported_type_has_error_trace_and_safe_detail(
    client: TestClient,
) -> None:
    files = {"file": ("x.png", io.BytesIO(b"\x89PNG\r\n"), "image/png")}
    response = client.post("/analyze/file", files=files)
    assert response.status_code == 415
    body = response.json()
    assert body["status"] == "error"
    assert body["error"]["code"] == "UNSUPPORTED_FILE"
    assert body["error"]["retryable"] is False
    assert body["trace"]["steps"]
    assert len(body["trace"]["steps"]) >= 1
    assert body["error"]["detail"] is None


def test_analyze_file_empty_extracted_text(client: TestClient) -> None:
    files = {"file": ("empty.txt", io.BytesIO(b"   \n"), "text/plain")}
    response = client.post("/analyze/file", files=files)
    assert response.status_code == 422
    body = response.json()
    assert body["status"] == "error"
    assert body["error"]["code"] == "TEXT_EXTRACTION_FAILED"
    assert body["trace"]["steps"]


def test_analyze_file_forwards_mode_to_pipeline(client: TestClient) -> None:
    files = {
        "file": (
            "doc.txt",
            io.BytesIO(b"Conteudo para verificar repasse do modo ao pipeline."),
            "text/plain",
        )
    }
    response = client.post(
        "/analyze/file",
        files=files,
        data={"regiao": "SP", "mode": "fast"},
    )
    assert response.status_code == 200
    mock_pipeline = client.app.state.pipeline
    mock_pipeline.analyze.assert_called_once()
    assert mock_pipeline.analyze.call_args.kwargs["mode"] == "fast"


def test_analyze_file_invalid_mode_becomes_standard(client: TestClient) -> None:
    files = {
        "file": (
            "doc.txt",
            io.BytesIO(b"Conteudo para modo invalido normalizado."),
            "text/plain",
        )
    }
    response = client.post(
        "/analyze/file",
        files=files,
        data={"mode": "turbo"},
    )
    assert response.status_code == 200
    assert (
        client.app.state.pipeline.analyze.call_args.kwargs["mode"] == "standard"
    )


def test_analyze_file_deep_adds_limitation_note(client: TestClient) -> None:
    files = {
        "file": (
            "doc.txt",
            io.BytesIO(b"Conteudo para modo profundo no parecer."),
            "text/plain",
        )
    }
    response = client.post("/analyze/file", files=files, data={"mode": "deep"})
    assert response.status_code == 200
    lims = response.json()["result"]["finalOpinion"]["limitations"]
    assert any("Modo profundo" in item for item in lims)


def test_analyze_file_fast_adds_limitation_note(client: TestClient) -> None:
    files = {
        "file": (
            "doc.txt",
            io.BytesIO(b"Conteudo para modo rapido no parecer."),
            "text/plain",
        )
    }
    response = client.post("/analyze/file", files=files, data={"mode": "fast"})
    assert response.status_code == 200
    lims = response.json()["result"]["finalOpinion"]["limitations"]
    assert any("Modo rápido" in item for item in lims)


def test_analyze_file_pipeline_failure_no_internal_leak(client: TestClient) -> None:
    mock_p = MagicMock()
    mock_p.analyze.side_effect = RuntimeError("secret_stack_token_xyz")
    client.app.state.pipeline = mock_p
    files = {
        "file": (
            "doc.txt",
            io.BytesIO(b"Texto suficiente para passar na etapa de extracao."),
            "text/plain",
        )
    }
    response = client.post("/analyze/file", files=files)
    assert response.status_code == 500
    body = response.json()
    assert body["error"]["code"] == "UNKNOWN"
    assert body["error"]["detail"] is None
    assert "secret_stack_token_xyz" not in response.text
