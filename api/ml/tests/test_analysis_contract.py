from __future__ import annotations

from unittest.mock import MagicMock

from api.main import (
    _build_analysis_response,
    _normalize_risk_level,
    _normalize_severity,
    _normalize_trace_steps,
    app,
)
from api.ml.pipeline import (
    AnalysisResult,
    AttentionPoint,
    ComplianceBlock,
    ComplianceItem,
    EntitiesBlock,
    SimilarCase,
)
from api.schemas.analysis import AnalysisResponse


def _sample_result() -> AnalysisResult:
    return AnalysisResult(
        contract_type="Consumidor",
        document_kind="outro",
        risk_score=39,
        risk_level="médio",
        attention_points=[
            AttentionPoint(
                tipo="liability_limitation",
                severidade="média",
                descricao="Ausência de limitação de responsabilidade.",
                clausula_referencia="Trecho de evidência.",
            )
        ],
        executive_summary="Resumo executivo.",
        main_risks=["Risco principal."],
        recommendations=["Recomendação."],
        positive_points=["Ponto positivo."],
        win_prediction="ganhou",
        win_probability=0.92,
        win_confidence=0.95,
        outcome_probabilities={"ganhou": 0.92},
        fee_estimate_min=1000.0,
        fee_estimate_max=3000.0,
        fee_estimate_suggested=2000.0,
        entities=EntitiesBlock(pessoas=["JOÃO DA SILVA"], valores=["R$ 1.000,00"]),
        similar_cases=[
            SimilarCase(
                id="case-1",
                tribunal="TJSP",
                number="123",
                tipo="Consumidor",
                titulo="Caso similar",
                resumo="Resumo do caso.",
                outcome="desconhecido",
                similaridade=0.42,
            )
        ],
        compliance=[
            ComplianceBlock(
                regulation="LGPD",
                items=[
                    ComplianceItem(
                        id="lgpd_base_legal",
                        description="Indica base legal.",
                        status="ok",
                        evidence="base legal no art. 7",
                    )
                ],
            )
        ],
        processing_time_seconds=1.2,
    )


def test_normaliza_enums_para_contrato_backend() -> None:
    assert _normalize_risk_level("médio") == "MEDIO"
    assert _normalize_risk_level("crítico") == "CRITICO"
    assert _normalize_severity("média") == "medium"
    assert _normalize_severity("alta") == "high"


def test_trace_steps_tem_fallback_validavel() -> None:
    steps = _normalize_trace_steps([], 123)
    assert steps[0]["step"] == "pipeline"
    assert steps[0]["provider"] == "internal"
    assert steps[0]["durationMs"] == 123


def test_build_analysis_response_valida_schema_e_preserva_similar_case() -> None:
    mock_p = MagicMock()
    mock_p.last_steps = []
    mock_p.last_external_trace = {}
    app.state.pipeline = mock_p
    payload = _build_analysis_response(
        result=_sample_result(),
        job_id="job-1",
        contract_id="contract-1",
        started_at="2026-04-20T00:00:00+00:00",
        finished_at="2026-04-20T00:00:01+00:00",
        duration_ms=1000,
        mode="standard",
    )
    parsed = AnalysisResponse.model_validate(payload)
    assert parsed.status == "done"
    assert parsed.result.risk.level == "MEDIO"
    assert parsed.result.attentionPoints[0].severity == "medium"
    assert parsed.result.similarCases[0].tribunal == "TJSP"
    assert parsed.result.similarCases[0].number == "123"
    assert parsed.result.outcomeProbability.value == 0.92
    assert parsed.result.outcomeProbability.rationale == "ganhou"
    assert parsed.result.outcomeProbability.confidence == 0.95
    assert parsed.result.urgency is not None
    assert parsed.result.urgency.level == "BAIXO"
    assert parsed.result.compliance[0].regulation == "LGPD"
    assert parsed.result.compliance[0].items[0].status == "ok"
    assert parsed.result.finalOpinion is not None
    assert parsed.result.finalOpinion.mainRisks == ["Risco principal."]
    assert parsed.result.finalOpinion.recommendations == ["Recomendação."]
    assert parsed.result.finalOpinion.positivePoints == ["Ponto positivo."]


def test_build_analysis_response_garante_blocos_obrigatorios_do_parecer() -> None:
    result = _sample_result()
    result.main_risks = []
    result.recommendations = []
    result.positive_points = []
    mock_p = MagicMock()
    mock_p.last_steps = []
    mock_p.last_external_trace = {}
    app.state.pipeline = mock_p

    payload = _build_analysis_response(
        result=result,
        job_id="job-1",
        contract_id="contract-1",
        started_at="2026-04-20T00:00:00+00:00",
        finished_at="2026-04-20T00:00:01+00:00",
        duration_ms=1000,
        mode="standard",
    )

    parsed = AnalysisResponse.model_validate(payload)
    assert parsed.result.finalOpinion is not None
    assert parsed.result.finalOpinion.mainRisks
    assert parsed.result.finalOpinion.recommendations
    assert parsed.result.finalOpinion.positivePoints


def test_build_analysis_response_garante_previsao_de_desfecho() -> None:
    result = _sample_result()
    result.win_prediction = ""
    result.win_probability = 1.7
    result.win_confidence = -0.2
    mock_p = MagicMock()
    mock_p.last_steps = []
    mock_p.last_external_trace = {}
    app.state.pipeline = mock_p

    payload = _build_analysis_response(
        result=result,
        job_id="job-1",
        contract_id="contract-1",
        started_at="2026-04-20T00:00:00+00:00",
        finished_at="2026-04-20T00:00:01+00:00",
        duration_ms=1000,
        mode="standard",
    )

    parsed = AnalysisResponse.model_validate(payload)
    assert parsed.result.outcomeProbability.rationale == "inconclusivo"
    assert parsed.result.outcomeProbability.value == 1.0
    assert parsed.result.outcomeProbability.confidence == 0.0
