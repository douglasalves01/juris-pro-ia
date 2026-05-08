from __future__ import annotations

from api.services.obligations_webhook import notify_obligations


def test_notify_obligations_sem_url_retorna_false() -> None:
    assert notify_obligations(None, job_id="j1", contract_id="c1", obligations=[{"subject": "x"}]) is False


def test_notify_obligations_envia_payload(monkeypatch) -> None:
    calls = []

    class Response:
        def raise_for_status(self) -> None:
            return None

    def fake_post(url: str, json: dict, timeout: float) -> Response:
        calls.append((url, json, timeout))
        return Response()

    monkeypatch.setattr("api.services.obligations_webhook.httpx.post", fake_post)

    result = notify_obligations(
        "https://backend.local/calendar",
        job_id="j1",
        contract_id="c1",
        obligations=[{"subject": "contratada", "obligation": "entregar"}],
    )

    assert result is True
    assert calls[0][0] == "https://backend.local/calendar"
    assert calls[0][1]["jobId"] == "j1"
    assert calls[0][1]["obligations"]
