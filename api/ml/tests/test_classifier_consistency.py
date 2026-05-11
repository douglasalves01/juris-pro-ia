from __future__ import annotations

import pytest

from api.ml.models import classifier


def test_classificacao_tipo_incompativel_usa_fallback_de_keywords() -> None:
    result = classifier.predict(
        "Reclamação trabalhista com pedido de horas extras, FGTS e aplicação da CLT.",
        "hf_models",
    )

    assert result["contract_type"] == "Trabalhista"
    assert result["classification_source"] == "keyword_fallback"
    assert "incompatível" in result["warning"]
    assert pytest.approx(sum(result["probs"].values()), rel=1e-6) == 1.0


def test_classificacao_tipo_fallback_sem_indicador_retorna_outros() -> None:
    result = classifier.predict("Documento genérico sem marcadores jurídicos específicos.", "hf_models")

    assert result["contract_type"] == "Outros"
    assert result["classification_source"] == "keyword_fallback"
    assert pytest.approx(sum(result["probs"].values()), rel=1e-6) == 1.0
