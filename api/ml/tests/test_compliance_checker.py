from __future__ import annotations

from hypothesis import given, settings, strategies as st

from api.ml.models import compliance_checker


def test_select_regulations_detecta_lgpd_por_texto() -> None:
    selected = compliance_checker.select_regulations(
        "Contrato trata dados pessoais conforme LGPD e privacidade.",
        "Tecnologia",
        "contrato",
    )

    assert selected
    assert selected[0]["regulation"] == "LGPD"


def test_checklist_lgpd_marca_item_ok_quando_base_legal_existe() -> None:
    result = compliance_checker.check(
        "O tratamento de dados pessoais observa a LGPD, possui base legal no art. 7 "
        "e adota medidas tecnicas de seguranca e confidencialidade.",
        "Tecnologia",
        "contrato",
    )

    lgpd = next(item for item in result if item.regulation == "LGPD")
    statuses = {item.id: item.status for item in lgpd.items}
    assert statuses["lgpd_base_legal"] == "ok"
    assert statuses["lgpd_segurança"] == "ok"


def test_checklist_cpc_detecta_peticao_inicial() -> None:
    result = compliance_checker.check(
        "Peticao inicial. Autor qualificado. DOS FATOS. DO DIREITO. DOS PEDIDOS. Valor da causa R$ 1000.",
        "Cível",
        "peticao_inicial",
    )

    assert any(item.regulation == "CPC" for item in result)


@given(text=st.text(max_size=1000), contract_type=st.text(max_size=80), document_kind=st.text(max_size=40))
@settings(max_examples=100)
def test_compliance_check_nunca_retorna_status_invalido(
    text: str,
    contract_type: str,
    document_kind: str,
) -> None:
    result = compliance_checker.check(text, contract_type, document_kind)

    for block in result:
        assert block.regulation
        for item in block.items:
            assert item.id
            assert item.description
            assert item.status in {"ok", "fail", "warning", "na"}
