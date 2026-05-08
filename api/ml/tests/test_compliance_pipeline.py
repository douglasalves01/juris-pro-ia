from __future__ import annotations

from api.ml import pipeline as pipeline_module
from api.ml.pipeline import AnalysisPipeline


def test_pipeline_deep_inclui_compliance_e_step(monkeypatch) -> None:
    monkeypatch.setenv("JURISPRO_SKIP_PRELOAD", "1")
    AnalysisPipeline._instance = None

    monkeypatch.setattr(
        pipeline_module.classifier,
        "predict_multi_chunk",
        lambda *args, **kwargs: {"contract_type": "Tecnologia", "probs": {"Tecnologia": 0.9}},
    )
    monkeypatch.setattr(
        pipeline_module.win_predictor,
        "predict",
        lambda *args, **kwargs: {
            "win_prediction": "inconclusivo",
            "win_probability": 0.5,
            "win_confidence": 0.5,
            "outcome_probabilities": {"inconclusivo": 0.5},
        },
    )
    monkeypatch.setattr(
        pipeline_module.risk_analyzer,
        "predict",
        lambda *args, **kwargs: {"risk_level": "baixo", "risk_score": 20, "attention_points": [], "model_probs": {"baixo": 0.8}},
    )
    monkeypatch.setattr(
        pipeline_module.ner,
        "predict",
        lambda *args, **kwargs: {"entities": {"pessoas": [], "organizacoes": [], "legislacao": ["LGPD"], "datas": [], "valores": []}},
    )
    monkeypatch.setattr(
        pipeline_module.summarizer,
        "predict",
        lambda *args, **kwargs: {"executive_summary": "Resumo."},
    )
    monkeypatch.setattr(
        pipeline_module.fee_estimator,
        "predict",
        lambda *args, **kwargs: {"fee_estimate_min": 1.0, "fee_estimate_max": 2.0, "fee_estimate_suggested": 1.5},
    )
    monkeypatch.setattr(
        pipeline_module.case_retriever,
        "predict",
        lambda *args, **kwargs: {"similar_cases": [], "similar_cases_notice": None},
    )

    pipe = AnalysisPipeline("hf_models")
    result = pipe.analyze(
        "Contrato de tecnologia com tratamento de dados pessoais conforme LGPD, "
        "base legal no art. 7, finalidade especifica e medidas tecnicas de seguranca.\n\n"
        "A contratada deve entregar o relatorio final no prazo de 30 dias.\n\n"
        "CLÁUSULA 1 - Multa\n"
        "Em caso de atraso, aplica-se multa de 35% sobre o valor total do contrato.",
        mode="deep",
    )

    assert result.compliance
    assert result.compliance[0].regulation == "LGPD"
    assert any(step["step"] == "compliance_check" for step in pipe.last_steps)
    assert any(step["step"] == "classify_clauses" for step in pipe.last_steps)
    assert any(step["step"] == "extract_obligations" for step in pipe.last_steps)
    assert any(point.referencia_tipo == "clause_classifier" for point in result.attention_points)
    assert result.obligations
