from __future__ import annotations

from unittest.mock import patch

from fastapi.testclient import TestClient
from hypothesis import given, settings, strategies as st

from api.main import app
from api.schemas.analysis import CounterArgumentsResponse
from api.services.counter_arguments_service import (
    build_from_attention_points,
    generate_counter_arguments,
)


_POINTS = st.lists(
    st.fixed_dictionaries(
        {
            "description": st.text(min_size=0, max_size=200),
            "severity": st.sampled_from(["low", "medium", "high", "critical", "baixa", "média", "alta"]),
            "clause": st.sampled_from(
                [
                    "lgpd_compliance",
                    "penalty_clause",
                    "termination_clause",
                    "jurisdiction_clause",
                    "liability_limitation",
                    "intellectual_property",
                    "dispute_resolution",
                    "outro",
                ]
            ),
        }
    ),
    max_size=30,
)


_NON_EMPTY_TEXT = st.text(min_size=1, max_size=1000).filter(lambda value: value.strip() != "")


@given(points=_POINTS, max_arguments=st.integers(min_value=1, max_value=20))
@settings(max_examples=100)
def test_fallback_preserva_campos_obrigatorios(points: list[dict], max_arguments: int) -> None:
    result = build_from_attention_points(points, max_arguments)

    assert len(result) <= max_arguments
    for item in result:
        assert item["text"].strip()
        assert item["strength"] in {"forte", "medio", "fraco"}
        assert item["category"].strip()


@given(points=_POINTS, max_arguments=st.integers(min_value=1, max_value=20))
@settings(max_examples=100)
def test_max_arguments_e_respeitado(points: list[dict], max_arguments: int) -> None:
    result = generate_counter_arguments(
        "Texto jurídico.",
        points,
        max_arguments,
        None,
        "https://example.invalid/v1",
        "model",
    )

    assert len(result) <= max_arguments


@given(text=_NON_EMPTY_TEXT, max_arguments=st.integers(min_value=1, max_value=20))
@settings(max_examples=100)
def test_counter_arguments_endpoint_entrada_valida_produz_schema(
    text: str,
    max_arguments: int,
) -> None:
    with (
        patch(
            "api.main.generate_counter_arguments",
            side_effect=lambda text, attention_points, max_arguments, api_key, base_url, model: [
                {"text": "Argumento adversarial.", "strength": "medio", "category": "Geral"}
            ][:max_arguments],
        ),
        TestClient(app) as client,
    ):
        response = client.post(
            "/analyze/counter-arguments",
            json={"text": text, "maxArguments": max_arguments},
        )

    assert response.status_code == 200
    parsed = CounterArgumentsResponse.model_validate(response.json())
    assert len(parsed.arguments) <= max_arguments


def test_counter_arguments_endpoint_retorna_schema(monkeypatch) -> None:
    monkeypatch.setattr(
        "api.main.generate_counter_arguments",
        lambda text, attention_points, max_arguments, api_key, base_url, model: [
            {"text": "Argumento adversarial.", "strength": "medio", "category": "Geral"}
        ],
    )

    with TestClient(app) as client:
        response = client.post(
            "/analyze/counter-arguments",
            json={"text": "Texto jurídico para análise.", "maxArguments": 3},
        )

    assert response.status_code == 200
    parsed = CounterArgumentsResponse.model_validate(response.json())
    assert len(parsed.arguments) == 1


def test_counter_arguments_endpoint_sem_entrada_retorna_422() -> None:
    with TestClient(app) as client:
        response = client.post("/analyze/counter-arguments", json={})

    assert response.status_code == 422


def test_counter_arguments_endpoint_rejeita_max_arguments_fora_do_intervalo() -> None:
    with TestClient(app) as client:
        low = client.post("/analyze/counter-arguments", json={"text": "x", "maxArguments": 0})
        high = client.post("/analyze/counter-arguments", json={"text": "x", "maxArguments": 21})

    assert low.status_code == 422
    assert high.status_code == 422


def test_counter_arguments_sem_attention_points_e_sem_llm_retorna_lista_vazia() -> None:
    result = generate_counter_arguments(
        "Texto sem análise prévia.",
        [],
        5,
        None,
        "https://example.invalid/v1",
        "model",
    )

    assert result == []
