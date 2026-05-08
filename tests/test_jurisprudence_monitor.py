from __future__ import annotations

import json

from fastapi.testclient import TestClient

from api.main import app, process_monitor_decisions
from api.schemas.analysis import MonitorAlertsResponse, MonitorSubscribeResponse
from api.services.jurisprudence_monitor import build_alerts, lexical_similarity
from scripts.ingest_jurisprudencia import ingest


def test_lexical_similarity_identifica_textos_relacionados() -> None:
    left = "plano de saude negativa de cobertura tratamento oncologico"
    right = "decisao sobre plano de saude e cobertura de tratamento oncologico"

    assert lexical_similarity(left, right) > 0.4


def test_build_alerts_respeita_threshold() -> None:
    alerts = build_alerts(
        {"case-1": {"referenceText": "plano de saude cobertura oncologico", "threshold": 0.3}},
        [{"id": "d1", "tribunal": "STJ", "summary": "plano de saude cobertura oncologico negada"}],
    )

    assert alerts
    assert alerts[0]["caseId"] == "case-1"


def test_monitor_endpoints_subscribe_and_list_alerts() -> None:
    with TestClient(app) as client:
        subscribe = client.post(
            "/monitor/subscribe",
            json={"caseId": "case-monitor-1", "contractId": "plano saude cobertura", "threshold": 0.2},
        )
        assert subscribe.status_code == 200
        parsed = MonitorSubscribeResponse.model_validate(subscribe.json())
        assert parsed.caseId == "case-monitor-1"

        process_monitor_decisions(
            [{"id": "dec-1", "tribunal": "STJ", "summary": "plano saude cobertura tratamento"}]
        )

        response = client.get("/monitor/alerts/case-monitor-1")
        assert response.status_code == 200
        alerts = MonitorAlertsResponse.model_validate(response.json())
        assert alerts.alerts
        assert alerts.alerts[0].newDecisionId == "dec-1"


def test_ingest_jurisprudencia_normaliza_json(tmp_path) -> None:
    source = tmp_path / "decisions.json"
    output = tmp_path / "out.jsonl"
    source.write_text(
        json.dumps([{"id": "d1", "tribunal": "STF", "ementa": "Resumo da decisao"}]),
        encoding="utf-8",
    )

    count = ingest(str(source), output)

    assert count == 1
    assert "Resumo da decisao" in output.read_text(encoding="utf-8")
