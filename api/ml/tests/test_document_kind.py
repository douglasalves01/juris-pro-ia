"""Testes leves para detecção de gênero documental."""

from __future__ import annotations

from api.ml.document_kind import detect_document_kind


def test_detect_peticao():
    t = """
    EXCELENTÍSSIMO SENHOR DOUTOR JUIZ
    DOS FATOS
    A parte autora vem propor ação de cobrança.
    DOS PEDIDOS
    Requer a citação.
    """
    assert detect_document_kind(t) == "peticao_inicial"


def test_detect_contrato():
    t = """
    CONTRATO DE PRESTAÇÃO DE SERVIÇOS
    CLÁUSULA PRIMEIRA — DO OBJETO
    O CONTRATANTE contrata os serviços da CONTRATADA.
    """
    assert detect_document_kind(t) == "contrato"


def test_short_text_outro():
    assert detect_document_kind("abc") == "outro"
