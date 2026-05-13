from __future__ import annotations

from api.ml import external_llm
from api.ml.external_llm import (
    _call_gemini,
    _extract_json_object,
    complete_analysis_fallback,
    should_invoke_external_llm,
)


def test_fast_never_invokes() -> None:
    assert not should_invoke_external_llm("fast", True, "sk-xxx")
    assert not should_invoke_external_llm("fast", False, None)


def test_standard_requires_gate_and_key() -> None:
    assert not should_invoke_external_llm("standard", False, "sk-xxx")
    assert not should_invoke_external_llm("standard", True, None)
    assert not should_invoke_external_llm("standard", True, "   ")
    assert should_invoke_external_llm("standard", True, "sk-xxx")


def test_deep_invokes_with_key_only() -> None:
    assert not should_invoke_external_llm("deep", False, None)
    assert should_invoke_external_llm("deep", False, "sk-xxx")


def test_complete_analysis_fallback_parseia_json_do_gemini(monkeypatch) -> None:
    call_kwargs = {}

    def fake_call(*args, **kwargs):
        call_kwargs.update(kwargs)
        return (
            """
            ```json
            {
              "executive_summary": "Resumo via fallback.",
              "main_risks": ["Risco processual relevante."],
              "recommendations": ["Revisar provas antes da distribuição."],
              "positive_points": ["Pedidos identificáveis no texto."],
              "outcome_probability": {
                "value": 0.61,
                "rationale": "probabilidade moderada de êxito",
                "confidence": 0.7
              }
            }
            ```
            """,
            100,
            50,
        )

    monkeypatch.setattr(external_llm, "_call_gemini", fake_call)

    result, _ = complete_analysis_fallback(
        api_key="key",
        model="gemini-2.0-flash",
        missing_fields=["outcome_probability", "main_risks"],
        executive_summary="",
        main_risks=[],
        recommendations=[],
        positive_points=[],
        contract_type="Consumidor",
        risk_level="médio",
        risk_score=50,
        win_prediction="inconclusivo",
        win_probability=0.33,
        document_kind="peticao_inicial",
        excerpt="Autor pede indenização por dano moral.",
    )

    assert result.used is True
    assert result.executive_summary == "Resumo via fallback."
    assert result.main_risks == ["Risco processual relevante."]
    assert result.recommendations == ["Revisar provas antes da distribuição."]
    assert result.positive_points == ["Pedidos identificáveis no texto."]
    assert result.outcome_probability == 0.61
    assert result.outcome_rationale == "probabilidade moderada de êxito"
    assert result.outcome_confidence == 0.7
    assert call_kwargs["response_mime_type"] == "application/json"
    assert call_kwargs["response_schema"]["type"] == "OBJECT"


def test_extract_json_object_aceita_texto_com_json_aninhado() -> None:
    parsed = _extract_json_object(
        """
        Claro, segue:
        {
          "main_risks": ["Risco A"],
          "outcome_probability": {
            "value": 0.42,
            "rationale": "incerto",
            "confidence": 0.51
          }
        }
        Observação final.
        """
    )

    assert parsed["main_risks"] == ["Risco A"]
    assert parsed["outcome_probability"]["value"] == 0.42


def test_complete_analysis_fallback_aproveita_texto_quando_json_invalido(monkeypatch) -> None:
    def fake_call(*args, **kwargs):
        return (
            "Há risco probatório relevante e o desfecho é incerto sem documentos adicionais.",
            100,
            50,
        )

    monkeypatch.setattr(external_llm, "_call_gemini", fake_call)

    result, _ = complete_analysis_fallback(
        api_key="key",
        model="gemini-2.5-flash",
        missing_fields=["outcome_probability", "main_risks"],
        executive_summary="Resumo.",
        main_risks=[],
        recommendations=[],
        positive_points=[],
        contract_type="Consumidor",
        risk_level="baixo",
        risk_score=29,
        win_prediction="inconclusivo",
        win_probability=0.0006,
        document_kind="peticao_inicial",
        excerpt="Autor pede indenização.",
    )

    assert result.used is True
    assert result.main_risks == [
        "Há risco probatório relevante e o desfecho é incerto sem documentos adicionais."
    ]
    assert result.outcome_probability == 0.5
    assert result.outcome_confidence == 0.4


def test_call_gemini_retries_em_429(monkeypatch) -> None:
    monkeypatch.setattr(external_llm.time, "sleep", lambda *_args, **_kwargs: None)
    calls = {"count": 0}

    class FakeResponse:
        def __init__(self, status_code: int) -> None:
            self.status_code = status_code
            self.headers = {}

        def raise_for_status(self) -> None:
            if self.status_code >= 400:
                request = external_llm.httpx.Request("POST", "https://gemini.test")
                response = external_llm.httpx.Response(self.status_code, request=request)
                raise external_llm.httpx.HTTPStatusError("erro", request=request, response=response)

        def json(self):
            return {
                "candidates": [{"content": {"parts": [{"text": "Resposta Gemini"}]}}],
                "usageMetadata": {"promptTokenCount": 12, "candidatesTokenCount": 8},
            }

    class FakeClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args) -> None:
            pass

        def post(self, *args, **kwargs):
            calls["count"] += 1
            return FakeResponse(429 if calls["count"] == 1 else 200)

    monkeypatch.setattr(external_llm.httpx, "Client", FakeClient)

    content, input_tokens, output_tokens = _call_gemini(
        "prompt",
        "key",
        "gemini-2.0-flash",
        1.0,
        retry_base_delay_sec=0.0,
    )

    assert calls["count"] == 2
    assert content == "Resposta Gemini"
    assert input_tokens == 12
    assert output_tokens == 8
