from __future__ import annotations

import uuid

from fastapi.testclient import TestClient

from api.main import app
from api.models.user import UserRole
from api.services.auth_service import create_access_token
from api.services import private_knowledge


def _token(firm_id: uuid.UUID) -> str:
    return create_access_token(
        uuid.uuid4(),
        "advogado@example.com",
        firm_id,
        UserRole.advogado,
    )


def test_private_knowledge_ingest_stats_and_draft_few_shot(monkeypatch) -> None:
    monkeypatch.setenv("JURISPRO_PRIVATE_RAG_BACKEND", "memory")
    private_knowledge.clear()
    firm_id = uuid.uuid4()
    headers = {"Authorization": f"Bearer {_token(firm_id)}"}

    with TestClient(app) as client:
        response = client.post(
            f"/firms/{firm_id}/knowledge/ingest",
            headers=headers,
            json={
                "documents": [
                    {
                        "documentId": str(uuid.uuid4()),
                        "type": "modelo_interno",
                        "title": "Modelo de negativacao indevida",
                        "text": "Pedido de danos morais por negativacao indevida com tutela de urgencia.",
                    }
                ]
            },
        )
        assert response.status_code == 200
        assert response.json()["status"] == "queued"

        stats = client.get(f"/firms/{firm_id}/knowledge/stats", headers=headers)
        assert stats.status_code == 200
        assert stats.json()["documentsIndexed"] == 1

        draft = client.post(
            "/generate/draft",
            json={
                "documentType": "peticao_inicial",
                "firmId": str(firm_id),
                "context": {
                    "parties": "Autor e Banco",
                    "subject": "Negativacao indevida",
                    "facts": "Inscricao indevida em cadastro restritivo.",
                    "claims": "Danos morais e tutela de urgencia.",
                },
            },
        )
        assert draft.status_code == 200
        assert "REFERENCIAS PRIVADAS DO ESCRITORIO" in draft.json()["draft"]


def test_private_knowledge_rejects_other_firm_token() -> None:
    firm_id = uuid.uuid4()
    other_firm = uuid.uuid4()

    with TestClient(app) as client:
        response = client.get(
            f"/firms/{firm_id}/knowledge/stats",
            headers={"Authorization": f"Bearer {_token(other_firm)}"},
        )

    assert response.status_code == 403
