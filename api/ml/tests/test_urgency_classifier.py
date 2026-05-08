from __future__ import annotations

from datetime import datetime, timedelta, timezone

from fastapi.testclient import TestClient
from hypothesis import given, settings, strategies as st

from api.main import app
from api.ml.pipeline import AnalysisResult, EntitiesBlock
from api.ml.models.urgency_classifier import (
    _KEYWORDS_IMEDIATO,
    _parse_date,
    classify,
)
from api.schemas.analysis import UrgencyResponse


_TEXT_PART = st.text(alphabet=st.characters(blacklist_categories=("Cs",)), max_size=120)


@given(
    prefix=_TEXT_PART,
    keyword=st.sampled_from(sorted(_KEYWORDS_IMEDIATO)),
    suffix=_TEXT_PART,
)
@settings(max_examples=100)
def test_keywords_imediatas_produzem_nivel_imediato(
    prefix: str,
    keyword: str,
    suffix: str,
) -> None:
    result = classify(f"{prefix} {keyword} {suffix}", [], "Processual")

    assert result.level == "IMEDIATO"
    assert result.score >= 80


@given(days=st.integers(min_value=0, max_value=7))
@settings(max_examples=100)
def test_datas_ate_7_dias_produzem_nivel_imediato(days: int) -> None:
    target = datetime.now(timezone.utc).date() + timedelta(days=days)

    result = classify("Texto sem palavras especiais " * 20, [target.strftime("%d/%m/%Y")])

    assert result.level == "IMEDIATO"
    assert result.score >= 80


@given(days=st.integers(min_value=8, max_value=30))
@settings(max_examples=100)
def test_datas_entre_8_e_30_dias_produzem_nivel_urgente(days: int) -> None:
    target = datetime.now(timezone.utc).date() + timedelta(days=days)

    result = classify("Texto sem palavras especiais " * 20, [target.strftime("%d-%m-%Y")])

    assert result.level == "URGENTE"
    assert 50 <= result.score <= 79


@given(text=_TEXT_PART, datas=st.lists(_TEXT_PART, max_size=8), contract_type=_TEXT_PART)
@settings(max_examples=100)
def test_rationale_nunca_fica_vazio(
    text: str,
    datas: list[str],
    contract_type: str,
) -> None:
    result = classify(text, datas, contract_type)

    assert len(result.rationale.strip()) >= 10


def test_parse_date_aceita_data_por_extenso() -> None:
    parsed = _parse_date("15 de março de 2026")

    assert parsed is not None
    assert parsed.day == 15
    assert parsed.month == 3
    assert parsed.year == 2026


def test_texto_muito_curto_sem_indicadores_produz_baixo() -> None:
    result = classify("texto simples sem marcador", [], "")

    assert result.level == "BAIXO"
    assert result.score < 20


def test_texto_vazio_produz_baixo() -> None:
    result = classify("", [], "")

    assert result.level == "BAIXO"
    assert result.score < 20


def test_data_proxima_tem_prioridade_sobre_keyword_urgente() -> None:
    target = datetime.now(timezone.utc).date() + timedelta(days=2)

    result = classify("Existe prazo para manifestação.", [target.strftime("%d/%m/%Y")])

    assert result.level == "IMEDIATO"
    assert result.score >= 80


@given(text=_TEXT_PART)
@settings(max_examples=100)
def test_analysis_result_inclui_urgency_com_schema_valido(text: str) -> None:
    urgency = classify(text, [], "")
    result = AnalysisResult(
        contract_type="Processual",
        document_kind="peticao_inicial",
        risk_score=20,
        risk_level="baixo",
        attention_points=[],
        executive_summary="Resumo.",
        main_risks=[],
        recommendations=[],
        positive_points=[],
        win_prediction="inconclusivo",
        win_probability=0.5,
        win_confidence=0.5,
        outcome_probabilities={},
        fee_estimate_min=0.0,
        fee_estimate_max=0.0,
        fee_estimate_suggested=0.0,
        entities=EntitiesBlock(),
        similar_cases=[],
        urgency={
            "score": urgency.score,
            "level": urgency.level,
            "rationale": urgency.rationale,
        },
        processing_time_seconds=0.01,
    )

    assert 0 <= result.urgency.score <= 100
    assert result.urgency.level in {"IMEDIATO", "URGENTE", "NORMAL", "BAIXO"}
    assert result.urgency.rationale


def test_analyze_urgency_endpoint_retorna_schema_para_texto_valido() -> None:
    with TestClient(app) as client:
        response = client.post(
            "/analyze/urgency",
            json={"text": "Pedido de liminar com tutela de urgência para análise imediata."},
        )

    assert response.status_code == 200
    parsed = UrgencyResponse.model_validate(response.json())
    assert parsed.level == "IMEDIATO"
    assert parsed.generatedAt


def test_analyze_urgency_endpoint_rejeita_texto_vazio() -> None:
    with TestClient(app) as client:
        response = client.post("/analyze/urgency", json={"text": "   "})

    assert response.status_code == 422
