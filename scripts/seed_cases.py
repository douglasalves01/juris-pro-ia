#!/usr/bin/env python3
"""
Popula `cases` e `case_embeddings` com 20 casos (domínio público / referências institucionais),
cobrindo os 12 tipos de classificação.

Uso:
  DATABASE_URL=postgresql+asyncpg://... python scripts/seed_cases.py
  (usa get_database_url_sync() para inserção síncrona)
"""

from __future__ import annotations

import logging
import sys
import uuid
from pathlib import Path

import numpy as np
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api.config import get_settings
from api.ml.models._common import resolve_submodel_path
from api.models.cases import Case, CaseEmbedding
from workers.db_sync import SessionLocal

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CASES: list[dict[str, object]] = [
    {
        "title": "Licença de software e responsabilidade por falhas (CDC)",
        "summary": (
            "Discussão sobre vício do produto digital e suporte técnico em contrato de licença "
            "de software empresarial. Aplicação do Código de Defesa do Consumidor a relação "
            "de consumo envolvendo licenciamento de sistemas de gestão."
        ),
        "outcome": "parcial",
        "contract_type": "Tecnologia",
        "court": "STJ",
        "region": "DF",
        "year": 2019,
        "source_url": "https://www.stj.jus.br/SCON/julgador/",
    },
    {
        "title": "Prestação de serviços de limpeza e terceirização",
        "summary": (
            "Ação sobre descumprimento de contrato de prestação de serviços de limpeza e "
            "conservação predial, com pedido de indenização por danos materiais e rescisão."
        ),
        "outcome": "procedente",
        "contract_type": "Serviços",
        "court": "TJMG",
        "region": "MG",
        "year": 2018,
        "source_url": "https://www.tjmg.jus.br/",
    },
    {
        "title": "Fornecimento de insumos com cláusula de exclusividade",
        "summary": (
            "Contrato de fornecimento contínuo de matérias-primas com exclusividade territorial "
            "e revisão de preço por indexador. Inadimplemento e rescisão contratual."
        ),
        "outcome": "parcial",
        "contract_type": "Fornecimento",
        "court": "STJ",
        "region": "DF",
        "year": 2017,
        "source_url": "https://www.stj.jus.br/SCON/julgador/",
    },
    {
        "title": "Fornecimento hospitalar e registro ANVISA",
        "summary": (
            "Fornecimento de equipamentos médico-hospitalares com instalação e treinamento. "
            "Defeito de fabricação e responsabilidade solidária na cadeia de fornecimento."
        ),
        "outcome": "procedente",
        "contract_type": "Fornecimento",
        "court": "TJRS",
        "region": "RS",
        "year": 2022,
        "source_url": "https://www.tjrs.jus.br/",
    },
    {
        "title": "Franquia e royalties sobre faturamento",
        "summary": (
            "Contrato de franquia com cobrança de royalties e taxa de publicidade. "
            "Discussão sobre informações pré-contratuais e equilíbrio econômico-financeiro."
        ),
        "outcome": "improcedente",
        "contract_type": "Parceria",
        "court": "STJ",
        "region": "DF",
        "year": 2016,
        "source_url": "https://www.stj.jus.br/SCON/julgador/",
    },
    {
        "title": "Parceria para exploração de mercado e propriedade intelectual",
        "summary": (
            "Acordo de cooperação técnica e cessão onerosa de uso de marca em projeto conjunto. "
            "Ruptura unilateral e indenização por perdas e danos."
        ),
        "outcome": "parcial",
        "contract_type": "Parceria",
        "court": "TJPR",
        "region": "PR",
        "year": 2023,
        "source_url": "https://www.tjpr.jus.br/",
    },
    {
        "title": "Empreitada global e atraso de obra civil",
        "summary": (
            "Contrato de empreitada para construção de edifício comercial. Atraso na entrega, "
            "multa moratória e comprovação de caso fortuito/força maior."
        ),
        "outcome": "parcial",
        "contract_type": "Obras",
        "court": "STJ",
        "region": "DF",
        "year": 2015,
        "source_url": "https://www.stj.jus.br/SCON/julgador/",
    },
    {
        "title": "Reforma industrial e responsabilidade por vícios ocultos",
        "summary": (
            "Contrato de reforma de planta industrial com empreitada parcial. Vícios de execução "
            "e prazo de garantia decenal conforme normas técnicas."
        ),
        "outcome": "procedente",
        "contract_type": "Obras",
        "court": "TJSC",
        "region": "SC",
        "year": 2019,
        "source_url": "https://www.tjsc.jus.br/",
    },
    {
        "title": "Reclamação trabalhista por horas extras e adicional noturno",
        "summary": (
            "Pedido de pagamento de horas extras não registradas e adicional noturno em contrato "
            "de prestação de serviços terceirizados. Nexo de emprego e responsabilidade subsidiária."
        ),
        "outcome": "parcial",
        "contract_type": "Trabalhista",
        "court": "TST",
        "region": "DF",
        "year": 2020,
        "source_url": "https://www.tst.jus.br/",
    },
    {
        "title": "Reconhecimento de vínculo e verbas rescisórias",
        "summary": (
            "Ação trabalhista visando reconhecimento de vínculo empregatício e pagamento de "
            "FGTS, férias e 13º salário em contrato de prestação de serviços."
        ),
        "outcome": "procedente",
        "contract_type": "Trabalhista",
        "court": "TRT",
        "region": "SP",
        "year": 2021,
        "source_url": "https://www.tst.jus.br/",
    },
    {
        "title": "Revisão de benefício previdenciário e salário de contribuição",
        "summary": (
            "Ação contra INSS pleiteando revisão do salário de benefício e recálculo de "
            "aposentadoria por tempo de contribuição, com correção monetária."
        ),
        "outcome": "parcial",
        "contract_type": "Previdenciário",
        "court": "TRF",
        "region": "DF",
        "year": 2018,
        "source_url": "https://www.trf1.jus.br/",
    },
    {
        "title": "Auxílio-doença negado e tutela de urgência",
        "summary": (
            "Pedido de concessão de auxílio-doença com laudos médicos e negativa administrativa. "
            "Nexo causal e incapacidade laborativa para atividade habitual."
        ),
        "outcome": "procedente",
        "contract_type": "Previdenciário",
        "court": "TRF",
        "region": "RS",
        "year": 2022,
        "source_url": "https://www.trf4.jus.br/",
    },
    {
        "title": "Mandado de segurança contra cobrança de ICMS",
        "summary": (
            "Mandado de segurança impetrado contra exigência de ICMS em cadeia de fornecimento "
            "de energia e insumos, com discussão sobre não cumulatividade e ajuste fiscal."
        ),
        "outcome": "procedente",
        "contract_type": "Tributário",
        "court": "STJ",
        "region": "DF",
        "year": 2014,
        "source_url": "https://www.stj.jus.br/SCON/julgador/",
    },
    {
        "title": "Embargos à execução fiscal e prescrição intercorrente",
        "summary": (
            "Embargos à execução fiscal contestando certidão de dívida ativa e alegando "
            "prescrição intercorrente e excesso de execução."
        ),
        "outcome": "improcedente",
        "contract_type": "Tributário",
        "court": "TJBA",
        "region": "BA",
        "year": 2019,
        "source_url": "https://www.tjba.jus.br/",
    },
    {
        "title": "Plano de saúde e negativa de cobertura de procedimento",
        "summary": (
            "Ação consumerista contra operadora de plano de saúde por negativa de cobertura de "
            "cirurgia com indicação médica. Aplicação do CDC e rol da ANS."
        ),
        "outcome": "procedente",
        "contract_type": "Consumidor",
        "court": "STJ",
        "region": "DF",
        "year": 2017,
        "source_url": "https://www.stj.jus.br/SCON/julgador/",
    },
    {
        "title": "Revisão de contrato bancário e juros abusivos",
        "summary": (
            "Revisão de contrato de financiamento com discussão sobre taxa de juros, "
            "capitalização e encargos moratórios em contratos de adesão."
        ),
        "outcome": "parcial",
        "contract_type": "Consumidor",
        "court": "STJ",
        "region": "DF",
        "year": 2016,
        "source_url": "https://www.stj.jus.br/SCON/julgador/",
    },
    {
        "title": "Divórcio litigioso e partilha de bens",
        "summary": (
            "Ação de dissolução de casamento com partilha de bens adquiridos na constância da "
            "união e discussão sobre regime de bens e ocultação patrimonial."
        ),
        "outcome": "parcial",
        "contract_type": "Família",
        "court": "TJCE",
        "region": "CE",
        "year": 2020,
        "source_url": "https://www.tjce.jus.br/",
    },
    {
        "title": "Investigação de paternidade e alimentos",
        "summary": (
            "Ação de investigação de paternidade cumulada com fixação de pensão alimentícia "
            "para filho menor, com exame de DNA e capacidade econômica do genitor."
        ),
        "outcome": "procedente",
        "contract_type": "Família",
        "court": "TJPE",
        "region": "PE",
        "year": 2021,
        "source_url": "https://www.tjpe.jus.br/",
    },
    {
        "title": "Habeas corpus e constrangimento ilegal",
        "summary": (
            "Habeas corpus preventivo contra decisão que determinou medidas cautelares penais "
            "desproporcionais, com análise de justa causa e tipicidade da conduta."
        ),
        "outcome": "procedente",
        "contract_type": "Criminal",
        "court": "STJ",
        "region": "DF",
        "year": 2022,
        "source_url": "https://www.stj.jus.br/SCON/julgador/",
    },
    {
        "title": "Locação comercial e reajuste pelo IGP-M",
        "summary": (
            "Contrato de locação comercial de imóvel para fins de varejo com prazo determinado, "
            "reajuste anual pelo IGP-M e discussão sobre multa rescisória e benfeitorias."
        ),
        "outcome": "acordo",
        "contract_type": "Outros",
        "court": "TJSP",
        "region": "SP",
        "year": 2018,
        "source_url": "https://esaj.tjsp.jus.br/",
    },
]


def _load_model():
    from sentence_transformers import SentenceTransformer

    settings = get_settings()
    path = resolve_submodel_path(str(settings.models_dir), "embeddings")
    if path.is_dir() and (path / "config.json").exists():
        logger.info("Carregando modelo de embeddings em %s", path)
        return SentenceTransformer(str(path))
    raise RuntimeError(
        f"Modelo de embeddings não encontrado em {path}. "
        "Monte hf_models/embeddings (SentenceTransformer) ou ajuste MODELS_DIR."
    )


def _truncate_global_cases(session: Session) -> None:
    ids = list(session.scalars(select(Case.id).where(Case.firm_id.is_(None))))
    if ids:
        session.execute(delete(CaseEmbedding).where(CaseEmbedding.case_id.in_(ids)))
        session.execute(delete(Case).where(Case.id.in_(ids)))
    session.commit()


def main() -> None:
    settings = get_settings()
    logger.info("MODELS_DIR=%s", settings.models_dir)

    texts = [str(c["summary"]) for c in CASES]
    model = _load_model()
    emb = model.encode(texts, convert_to_numpy=True, show_progress_bar=True, normalize_embeddings=True)

    arr = np.asarray(emb, dtype=np.float32)
    if arr.shape != (len(CASES), 768):
        raise RuntimeError(f"Embeddings com shape {arr.shape}; esperado ({len(CASES)}, 768).")

    with SessionLocal() as session:
        _truncate_global_cases(session)
        for i, row in enumerate(CASES):
            cid = uuid.uuid4()
            case = Case(
                id=cid,
                firm_id=None,
                title=str(row["title"]),
                summary=str(row["summary"]),
                outcome=str(row["outcome"]),
                contract_type=str(row["contract_type"]),
                court=str(row["court"]),
                region=str(row["region"]),
                year=int(row["year"]),
                source_url=str(row["source_url"]),
            )
            session.add(case)
            session.add(CaseEmbedding(case_id=cid, embedding=arr[i].tolist()))
        session.commit()

    logger.info("Inseridos %s casos globais com embeddings.", len(CASES))


if __name__ == "__main__":
    main()
