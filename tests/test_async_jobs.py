"""Jobs assíncronos: 202 + polling."""

from __future__ import annotations

import io
import time
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from api.main import app
from api.ml.pipeline import AnalysisPipeline, AnalysisResult, EntitiesBlock

AnalysisPipeline._instance = None


def _minimal_result() -> AnalysisResult:
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
        mock_p.last_external_trace = {
            "used": False,
            "provider": None,
            "model": None,
            "cost_usd": 0.0,
        }
        mock_p.analyze.return_value = _minimal_result()
        tc.app.state.pipeline = mock_p
        yield tc


def test_async_returns_202_and_poll_finds_done(client: TestClient) -> None:
    files = {
        "file": (
            "doc.txt",
            io.BytesIO(b"Peticao com conteudo juridico para fila async."),
            "text/plain",
        )
    }
    response = client.post(
        "/analyze/file/async",
        files=files,
        data={"jobId": "job-e2e-1", "regiao": "SP"},
    )
    assert response.status_code == 202
    body = response.json()
    assert body["status"] == "queued"
    assert body["jobId"] == "job-e2e-1"
    assert "/jobs/job-e2e-1" in body.get("pollUrl", "")

    for _ in range(80):
        poll = client.get("/jobs/job-e2e-1")
        assert poll.status_code == 200
        payload = poll.json()
        if payload.get("status") == "done":
            assert payload.get("result") is not None
            assert payload["result"]["document"]["legalArea"] == "Consumidor"
            return
        time.sleep(0.03)
    raise AssertionError("timeout aguardando conclusão do job")


def test_async_celery_backend_enfileira_e_poll_retorna_done(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from workers.celery_app import celery_app

    old_always_eager = celery_app.conf.task_always_eager
    old_store_eager = celery_app.conf.task_store_eager_result
    celery_app.conf.task_always_eager = True
    celery_app.conf.task_store_eager_result = False
    monkeypatch.setenv("JURISPRO_QUEUE_BACKEND", "celery")
    try:
        files = {
            "file": (
                "doc.txt",
                io.BytesIO(b"Peticao com conteudo juridico para fila celery."),
                "text/plain",
            )
        }
        response = client.post(
            "/analyze/file/async",
            files=files,
            data={"jobId": "job-celery-eager", "regiao": "SP"},
        )
        assert response.status_code == 202
        body = response.json()
        assert body["queueBackend"] == "celery"

        poll = client.get("/jobs/job-celery-eager")
        assert poll.status_code == 200
        payload = poll.json()
        assert payload["status"] == "done"
        assert payload["result"]["document"]["legalArea"] == "Consumidor"
    finally:
        celery_app.conf.task_always_eager = old_always_eager
        celery_app.conf.task_store_eager_result = old_store_eager


def test_async_conflict_when_job_id_still_active(client: TestClient) -> None:
    def _slow_analyze(*_a: object, **_kw: object) -> AnalysisResult:
        time.sleep(0.35)
        return _minimal_result()

    client.app.state.pipeline.analyze.side_effect = _slow_analyze

    files = {
        "file": (
            "slow.txt",
            io.BytesIO(b"Conteudo para bloquear job duplicado."),
            "text/plain",
        )
    }
    first = client.post(
        "/analyze/file/async",
        files=files,
        data={"jobId": "job-dup"},
    )
    assert first.status_code == 202

    second = client.post(
        "/analyze/file/async",
        files=files,
        data={"jobId": "job-dup"},
    )
    assert second.status_code == 409
    err = second.json()
    assert err["status"] == "error"
    assert err["error"]["code"] == "JOB_ALREADY_ACTIVE"
    assert err["trace"]["steps"]


def test_job_unknown_returns_404(client: TestClient) -> None:
    response = client.get("/jobs/nao-existe-uuid")
    assert response.status_code == 404


def test_async_unsupported_file_returns_error_contract(client: TestClient) -> None:
    files = {"file": ("bad.bin", b"x", "application/octet-stream")}
    response = client.post(
        "/analyze/file/async",
        files=files,
        data={"jobId": "job-bad-ext", "contractId": "c1"},
    )
    assert response.status_code == 415
    body = response.json()
    assert body["status"] == "error"
    assert body["error"]["code"] == "UNSUPPORTED_FILE"
    assert body["jobId"] == "job-bad-ext"
    assert body["contractId"] == "c1"
    assert body["trace"]["steps"]
