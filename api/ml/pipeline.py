"""Orquestrador da análise jurídica multimodelo."""

from __future__ import annotations

import os
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

from typing import Literal

from pydantic import BaseModel, Field

from api.config import get_settings
from api.ml.document_kind import detect_document_kind
from api.ml import external_llm
from api.ml.models import (
    case_retriever,
    clause_classifier,
    classifier,
    compliance_checker,
    contract_differ,
    fee_estimator,
    ner,
    obligation_extractor,
    risk_analyzer,
    summarizer,
    urgency_classifier,
    win_predictor,
)
from api.ml.preprocessor import TextPreprocessor
from api.services import semantic_cache


class AttentionPoint(BaseModel):
    tipo: str
    severidade: str
    descricao: str
    clausula_referencia: str | None = None
    referencia_tipo: str = "trecho_processual"


class Clause(BaseModel):
    numero: str
    tipo: str
    titulo: str
    texto: str


class TimelineEvent(BaseModel):
    data: str
    evento: str


class SimilarCase(BaseModel):
    id: str
    tribunal: str = ""
    number: str = ""
    tipo: str = ""
    titulo: str
    resumo: str
    outcome: str
    similaridade: float


class EntitiesBlock(BaseModel):
    pessoas: list[str] = Field(default_factory=list)
    organizacoes: list[str] = Field(default_factory=list)
    legislacao: list[str] = Field(default_factory=list)
    datas: list[str] = Field(default_factory=list)
    valores: list[str] = Field(default_factory=list)


class UrgencyBlock(BaseModel):
    score: int = Field(ge=0, le=100)
    level: Literal["IMEDIATO", "URGENTE", "NORMAL", "BAIXO"]
    rationale: str


class ComplianceItem(BaseModel):
    id: str
    description: str
    status: Literal["ok", "fail", "warning", "na"]
    evidence: str | None = None


class ComplianceBlock(BaseModel):
    regulation: str
    items: list[ComplianceItem] = Field(default_factory=list)


class ObligationBlock(BaseModel):
    subject: str
    obligation: str
    deadline: str | None = None
    deadlineAbsolute: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)


class AnalysisResult(BaseModel):
    contract_type: str
    document_kind: str = "outro"
    risk_score: int = Field(ge=0, le=100)
    risk_level: str
    attention_points: list[AttentionPoint]
    executive_summary: str
    main_risks: list[str]
    recommendations: list[str]
    positive_points: list[str]
    win_prediction: str
    win_probability: float = Field(
        ge=0.0,
        le=1.0,
        description="Probabilidade estimada da classe 'ganhou' (distribuição completa em outcome_probabilities).",
    )
    win_confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confiança do modelo na classe indicada em win_prediction.",
    )
    outcome_probabilities: dict[str, float] = Field(default_factory=dict)
    fee_estimate_min: float
    fee_estimate_max: float
    fee_estimate_suggested: float
    entities: EntitiesBlock
    similar_cases: list[SimilarCase]
    similar_cases_notice: str | None = None
    clauses: list[Clause] = Field(default_factory=list)
    timeline: list[TimelineEvent] = Field(default_factory=list)
    section_summary: dict[str, str] = Field(default_factory=dict)
    urgency: UrgencyBlock = Field(
        default_factory=lambda: UrgencyBlock(score=0, level="BAIXO", rationale="Não classificado.")
    )
    compliance: list[ComplianceBlock] = Field(default_factory=list)
    obligations: list[ObligationBlock] = Field(default_factory=list)
    processing_time_seconds: float
    cache_hit: bool = False


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def resolve_models_dir(models_dir: str) -> Path:
    """Resolve diretório de pesos ML (preferência: hf_models/ na raiz do repositório)."""
    root = _project_root()
    p = Path(models_dir)
    if not p.is_absolute():
        p = root / p
    p = p.resolve()
    if (p / "ner_juridico").is_dir():
        return p
    hf = (root / "hf_models").resolve()
    if (hf / "ner_juridico").is_dir():
        return hf
    # Legado: modelos ainda soltos na raiz do projeto
    if (root / "ner_juridico").is_dir():
        return root
    return p


def _derive_main_risks(
    attention: list[AttentionPoint],
    risk_score: int,
    risk_level: str,
    document_kind: str,
) -> list[str]:
    ordered = sorted(
        attention,
        key=lambda a: {"alta": 3, "média": 2, "baixa": 1}.get(a.severidade, 0),
        reverse=True,
    )
    lines = [a.descricao for a in ordered if a.severidade in ("alta", "média", "crítica", "crítico")]
    if risk_score >= 60 and len(lines) < 3:
        lines.append(f"Pontuação global de risco elevada ({risk_score}/100), nível {risk_level}.")
    if not lines and document_kind == "peticao_inicial":
        lines.append(
            "Sem alertas automáticos de cláusulas contratuais nesta peça; priorizar mérito, "
            "provas e riscos processuais na revisão humana."
        )
    return lines[:15]


def _derive_recommendations(
    attention: list[AttentionPoint],
    risk_level: str,
    entities: EntitiesBlock,
    contract_type: str,
    document_kind: str,
) -> list[str]:
    recs: list[str] = []
    tipos = {a.tipo for a in attention}

    if document_kind == "peticao_inicial":
        recs.append(
            "Revisar cabeçalho, competência, valor da causa e documentos indispensáveis (art. 319 CPC)."
        )

    if "lgpd_compliance" in tipos:
        recs.append(
            "Formalizar bases legais (Art. 7/11 LGPD), papel do encarregado (DPO) e medidas de segurança."
        )
    if "penalty_clause" in tipos:
        recs.append("Rever percentuais de multa e proporcionalidade com o inadimplemento (CDC/CC).")
    if "termination_clause" in tipos:
        recs.append("Ajustar prazos de aviso prévio e reciprocidade na rescisão contratual.")
    if "jurisdiction_clause" in tipos:
        recs.append("Reavaliar eleição de foro e lei aplicável, com foco em defesa e custos processuais.")
    if "liability_limitation" in tipos:
        recs.append("Inserir teto de responsabilidade e carve-outs para dolo ou grave negligência.")
    if "intellectual_property" in tipos:
        recs.append("Definir titularidade, licenças e cessão de PI com clareza e registro quando couber.")
    if "dispute_resolution" in tipos and risk_level in ("alto", "crítico"):
        recs.append("Considerar mediação/arbitragem para contratos de alto valor e redução de litígio.")
    if "abusive_clause" in tipos:
        recs.append("Revisar cláusulas sinalizadas como potencialmente abusivas e ajustar proporcionalidade/reciprocidade.")

    if len(entities.legislacao) >= 3:
        if document_kind == "peticao_inicial":
            recs.append("Conferir citação e pertinência das normas invocadas em relação aos fatos e pedidos da peça.")
        else:
            recs.append("Conferir consistência entre normas citadas e o objeto do negócio jurídico.")

    if document_kind != "peticao_inicial" and contract_type in ("Tecnologia", "Serviços", "Parceria"):
        recs.append("Documentar SLAs, aceite e plano de contingência para entregas críticas.")
    elif document_kind == "peticao_inicial" and contract_type == "Tecnologia":
        recs.append(
            "Reunir provas documentais do escopo de software (e-mails, cronogramas, homologações, atas de entrega)."
        )

    if not recs:
        recs.append("Revisar cláusulas gerais e anexos técnicos com suporte jurídico especializado.")

    return recs[:18]


def _normalize_analysis_mode(mode: str | None) -> str:
    m = (mode or "standard").strip().lower()
    return m if m in {"fast", "standard", "deep"} else "standard"


def _external_llm_gate_triggered(
    risk_score: int,
    cls_conf: float | None,
    num_attention_signals: int,
) -> bool:
    """Gatilhos para uso futuro de LLM externo no modo standard (stub)."""
    if risk_score >= 70:
        return True
    if cls_conf is not None and cls_conf < 0.35:
        return True
    if num_attention_signals >= 6:
        return True
    return False


def _derive_positive_points(
    risk_score: int,
    risk_level: str,
    attention: list[AttentionPoint],
    entities: EntitiesBlock,
) -> list[str]:
    out: list[str] = []
    if risk_level == "baixo" and risk_score <= 40:
        out.append("Perfil de risco predominante baixo nas dimensões analisadas automaticamente.")
    if len(attention) <= 1 and risk_score <= 45:
        out.append("Poucos alertas heurísticos relevantes foram disparados no texto.")
    if entities.organizacoes:
        out.append(
            f"Organizações identificadas após filtragem ({len(entities.organizacoes)} entidade(s))."
        )
    if entities.legislacao:
        out.append("Referências normativas detectadas — base para análise de compliance.")
    if len(out) > 4:
        return out[:4]
    if not out:
        out.append("Texto processado integralmente; utilize os campos de risco e entidades para revisão.")
    return out


class AnalysisPipeline:
    """Singleton: carrega referências aos modelos uma única vez por processo."""

    _instance: AnalysisPipeline | None = None
    _singleton_lock = threading.Lock()

    def __new__(cls, models_dir: str = "hf_models") -> AnalysisPipeline:
        if cls._instance is None:
            with cls._singleton_lock:
                if cls._instance is None:
                    inst = super().__new__(cls)
                    inst._initialized = False
                    cls._instance = inst
        assert cls._instance is not None
        return cls._instance

    def __init__(self, models_dir: str = "hf_models") -> None:
        if getattr(self, "_initialized", False):
            return
        self._models_dir = resolve_models_dir(models_dir)
        self._preprocessor = TextPreprocessor()
        self._last_steps: list[dict[str, object]] = []
        self._last_external_used = False
        self._last_external_provider: str | None = None
        self._last_external_model: str | None = None
        self._last_external_cost_usd = 0.0
        self._preload_models()
        self._initialized = True

    @property
    def last_steps(self) -> list[dict[str, object]]:
        return list(getattr(self, "_last_steps", []))

    @property
    def last_external_trace(self) -> dict[str, object]:
        return {
            "used": bool(getattr(self, "_last_external_used", False)),
            "provider": getattr(self, "_last_external_provider", None),
            "model": getattr(self, "_last_external_model", None),
            "cost_usd": float(getattr(self, "_last_external_cost_usd", 0.0) or 0.0),
        }

    def _record_step(
        self,
        steps: list[dict[str, object]],
        step: str,
        started: float,
        provider: str = "internal",
        model: str | None = None,
        confidence: float | None = None,
        *,
        model_version: str | None = None,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        estimated_cost_usd: float = 0.0,
    ) -> None:
        now = time.perf_counter()
        duration_ms = max(0, int((now - started) * 1000))
        wall_now = datetime.now(timezone.utc)
        elapsed_sec = now - started
        started_at_iso = (wall_now - timedelta(seconds=elapsed_sec)).isoformat()
        finished_at_iso = wall_now.isoformat()
        step_payload = {
            "step": step,
            "provider": provider,
            "model": model,
            "modelVersion": model_version,
            "durationMs": duration_ms,
            "startedAt": started_at_iso,
            "finishedAt": finished_at_iso,
            "confidence": confidence,
            "inputTokens": input_tokens,
            "outputTokens": output_tokens,
            "estimatedCostUsd": estimated_cost_usd,
        }
        steps.append(step_payload)
        self._last_steps = list(steps)

    def _preload_models(self) -> None:
        """Dispara o carregamento lazy de cada módulo uma única vez."""
        if os.getenv("JURISPRO_SKIP_PRELOAD", "").strip().lower() in ("1", "true", "yes"):
            return
        root = str(self._models_dir)
        stub = (
            "Contrato de prestação de serviços. CLÁUSULA PRIMEIRA: multa de 25% sobre o valor. "
            "Tratamento de dados pessoais conforme LGPD. Rescisão unilateral em 15 dias. "
            "Valor R$ 120.000,00."
        )
        _ = ner.predict(stub, root)
        _ = classifier.predict(stub, root)
        _ = win_predictor.predict(stub, root)
        _ = summarizer.predict(stub, root, document_kind="contrato", sections={})
        _ = risk_analyzer.predict(stub, root, classification_text=stub, document_kind="contrato")
        _ = fee_estimator.predict(stub, root, "Outros", "SP", 40)
        _ = case_retriever.predict(stub, root, top_k=1)

    def analyze(
        self,
        text: str,
        regiao: str = "SP",
        firm_id: uuid.UUID | None = None,
        mode: str | None = "standard",
    ) -> AnalysisResult:
        mode = _normalize_analysis_mode(mode)
        t0 = time.perf_counter()
        steps: list[dict[str, object]] = []
        self._last_steps = []
        self._last_external_used = False
        self._last_external_provider = None
        self._last_external_model = None
        self._last_external_cost_usd = 0.0
        s0 = time.perf_counter()
        cleaned = self._preprocessor.clean(text or "")
        document_kind = detect_document_kind(cleaned)
        chunks = self._preprocessor.split_into_chunks(cleaned, max_tokens=450, overlap=50)
        rep = self._preprocessor.get_representative_chunk(chunks, max_tokens=512)
        if not rep.strip():
            rep = cleaned[:8000]
        self._record_step(steps, "chunk_document", s0, model="text-preprocessor", confidence=1.0)

        chunk_inputs = chunks if chunks else ([rep] if rep.strip() else [cleaned[:4000] or ""])
        models_dir = str(self._models_dir)
        dispositivo = self._preprocessor.extract_dispositivo(cleaned)
        win_text = dispositivo if dispositivo else cleaned
        parallel_executor = ThreadPoolExecutor(max_workers=5)
        s_cls = time.perf_counter()
        cls_future = parallel_executor.submit(
            classifier.predict_multi_chunk,
            chunk_inputs,
            models_dir,
            document_kind,
            rep + "\n\n" + cleaned[:6000],
        )
        s_win = time.perf_counter()
        win_future = parallel_executor.submit(win_predictor.predict, win_text, models_dir)
        s_risk = time.perf_counter()
        risk_future = parallel_executor.submit(
            risk_analyzer.predict,
            cleaned,
            models_dir,
            classification_text=rep,
            document_kind=document_kind,
        )
        s_ner = time.perf_counter()
        ner_future = parallel_executor.submit(ner.predict, cleaned, models_dir)
        top_k_similar = 8 if mode == "deep" else 5
        s_search = time.perf_counter()
        similar_future = None
        if mode != "fast":
            similar_future = parallel_executor.submit(
                case_retriever.predict,
                cleaned,
                models_dir,
                top_k=top_k_similar,
                firm_id=firm_id,
            )

        cls_out = cls_future.result()
        contract_type = str(cls_out.get("contract_type", "Outros"))
        cls_probs = cls_out.get("probs") or {}
        cls_conf = max((float(v) for v in cls_probs.values()), default=None)
        self._record_step(
            steps,
            "classify_document",
            s_cls,
            provider="huggingface",
            model="classificacao_tipo",
            confidence=cls_conf,
        )

        win_out = win_future.result()
        win_prediction = str(win_out.get("win_prediction", "inconclusivo"))
        win_probability = float(win_out.get("win_probability", 0.0))
        win_confidence = float(win_out.get("win_confidence", win_probability))
        outcome_probs = {str(k): float(v) for k, v in (win_out.get("outcome_probabilities") or win_out.get("probs") or {}).items()}
        self._record_step(
            steps,
            "predict_outcome",
            s_win,
            provider="huggingface",
            model="predicao_ganho",
            confidence=win_confidence,
        )

        risk_out = risk_future.result()
        risk_level = str(risk_out.get("risk_level", "médio"))
        risk_score = int(risk_out.get("risk_score", 50))
        risk_probs = risk_out.get("model_probs") or {}
        risk_conf = max((float(v) for v in risk_probs.values()), default=None)
        raw_points = risk_out.get("attention_points") or []
        attention_points = [
            AttentionPoint(
                tipo=str(p.get("tipo", "outro")),
                severidade=str(p.get("severidade", "média")),
                descricao=str(p.get("descricao", "")),
                clausula_referencia=p.get("clausula_referencia"),
                referencia_tipo=str(p.get("referencia_tipo") or "trecho_processual"),
            )
            for p in raw_points
            if isinstance(p, dict)
        ]
        self._record_step(
            steps,
            "risk_scoring",
            s_risk,
            provider="huggingface",
            model="analise_risco",
            confidence=risk_conf,
        )

        gate_on = _external_llm_gate_triggered(
            risk_score, cls_conf, len(raw_points)
        )
        if mode == "standard":
            s_gate = time.perf_counter()
            self._record_step(
                steps,
                "external_api_gate",
                s_gate,
                provider="internal",
                model="llm-trigger" if gate_on else "llm-not-needed",
                confidence=1.0 if gate_on else 0.95,
            )

        ner_out = ner_future.result()
        ent_dict = ner_out.get("entities") or {}
        entities = EntitiesBlock(
            pessoas=list(ent_dict.get("pessoas") or []),
            organizacoes=list(ent_dict.get("organizacoes") or []),
            legislacao=list(ent_dict.get("legislacao") or []),
            datas=list(ent_dict.get("datas") or []),
            valores=list(ent_dict.get("valores") or []),
        )
        self._record_step(
            steps,
            "extract_entities",
            s_ner,
            provider="huggingface",
            model="ner_juridico",
            confidence=0.8 if any(ent_dict.values()) else 0.5,
        )

        s0 = time.perf_counter()
        urgency_out = urgency_classifier.classify(
            cleaned,
            list(entities.datas),
            contract_type,
        )
        urgency = UrgencyBlock(
            score=int(urgency_out.score),
            level=urgency_out.level,  # type: ignore[arg-type]
            rationale=str(urgency_out.rationale),
        )
        self._record_step(
            steps,
            "urgency_classification",
            s0,
            provider="rules",
            model="urgency-rules",
            confidence=0.9,
        )

        s0 = time.perf_counter()
        obligations = [
            ObligationBlock(
                subject=item.subject,
                obligation=item.obligation,
                deadline=item.deadline,
                deadlineAbsolute=item.deadlineAbsolute,
                confidence=item.confidence,
            )
            for item in obligation_extractor.extract(cleaned, list(entities.datas))
        ]
        self._record_step(
            steps,
            "extract_obligations",
            s0,
            provider="rules",
            model="obligation-rules",
            confidence=0.75 if obligations else 0.45,
        )

        settings = get_settings()
        compliance: list[ComplianceBlock] = []
        if mode == "deep":
            s0 = time.perf_counter()
            compliance_raw = compliance_checker.check(
                cleaned,
                contract_type=contract_type,
                document_kind=document_kind,
                api_key=settings.openai_api_key,
                base_url=str(settings.openai_base_url),
                model=str(settings.openai_model),
            )
            compliance = [
                ComplianceBlock(
                    regulation=item.regulation,
                    items=[
                        ComplianceItem(
                            id=entry.id,
                            description=entry.description,
                            status=entry.status,
                            evidence=entry.evidence,
                        )
                        for entry in item.items
                    ],
                )
                for item in compliance_raw
            ]
            self._record_step(
                steps,
                "compliance_check",
                s0,
                provider="rules",
                model="compliance-checklists",
                confidence=0.75 if compliance else 0.5,
            )

        s0 = time.perf_counter()
        sections = self._preprocessor.extract_sections(cleaned)
        raw_clauses = self._preprocessor.extract_clauses(cleaned)
        clause_classifications = clause_classifier.classify_clauses(
            raw_clauses,
            str(self._models_dir),
        )
        for clause_out in clause_classifications:
            if clause_out.label != "abusiva":
                continue
            attention_points.append(
                AttentionPoint(
                    tipo="abusive_clause",
                    severidade="alta" if clause_out.confidence >= 0.8 else "média",
                    descricao=f"Cláusula potencialmente abusiva: {clause_out.rationale}",
                    clausula_referencia=clause_out.evidence,
                    referencia_tipo="clause_classifier",
                )
            )
        if raw_clauses:
            self._record_step(
                steps,
                "classify_clauses",
                s0,
                provider="rules",
                model="clause-classifier-rules",
                confidence=max((item.confidence for item in clause_classifications), default=0.0),
            )

        s0 = time.perf_counter()
        focus = self._preprocessor.extract_summary_focus_text(cleaned, document_kind).strip()
        if not focus:
            focus = rep
        sum_out = summarizer.predict(
            focus,
            str(self._models_dir),
            document_kind=document_kind,
            sections=sections,
            use_seq2seq=(mode != "fast"),
            deep=(mode == "deep"),
        )
        executive_summary = str(sum_out.get("executive_summary", "")).strip()
        if not executive_summary:
            executive_summary = rep[:600] + ("…" if len(rep) > 600 else "")
        self._record_step(
            steps,
            "summarize_document",
            s0,
            provider="huggingface",
            model="sumarizacao-extractive" if mode == "fast" else "sumarizacao",
            confidence=0.75 if executive_summary else 0.4,
        )

        s0 = time.perf_counter()
        fee_out = fee_estimator.predict(
            cleaned,
            str(self._models_dir),
            contract_type=contract_type,
            regiao=regiao,
            risk_score=risk_score,
        )
        self._record_step(
            steps,
            "estimate_fees",
            s0,
            provider="internal",
            model="honorarios",
            confidence=0.7,
        )

        if mode == "fast":
            similar_raw: list = []
            similar_notice = (
                "Modo rápido: busca por casos similares (Qdrant) não foi executada."
            )
            similar_cases = []
            self._record_step(
                steps,
                "embedding_search",
                s_search,
                provider="internal",
                model="skipped-fast-mode",
                confidence=None,
            )
        else:
            assert similar_future is not None
            sim_out = similar_future.result()
            similar_raw = sim_out.get("similar_cases") or []
            similar_notice = sim_out.get("similar_cases_notice")
            similar_cases = [
                SimilarCase(
                    id=str(s.get("id", "")),
                    tribunal=str(s.get("tribunal", "")),
                    number=str(s.get("number") or s.get("numeroProcesso") or ""),
                    tipo=str(s.get("tipo", "")),
                    titulo=str(s.get("titulo", "")),
                    resumo=str(s.get("resumo", "")),
                    outcome=str(s.get("outcome", "")),
                    similaridade=float(s.get("similaridade", 0.0)),
                )
                for s in similar_raw
                if isinstance(s, dict)
            ]
            self._record_step(
                steps,
                "embedding_search",
                s_search,
                provider="internal",
                model="qdrant+embeddings",
                confidence=max((s.similaridade for s in similar_cases), default=0.0),
            )
        parallel_executor.shutdown(wait=True)

        s0 = time.perf_counter()
        main_risks = _derive_main_risks(
            attention_points, risk_score, risk_level, document_kind
        )
        recommendations = _derive_recommendations(
            attention_points, risk_level, entities, contract_type, document_kind
        )
        positive_points = _derive_positive_points(
            risk_score, risk_level, attention_points, entities
        )
        self._record_step(steps, "validate_output", s0, model="rules", confidence=0.85)

        s_llm = time.perf_counter()
        should_try_llm = external_llm.should_invoke_external_llm(
            mode,
            gate_on,
            settings.openai_api_key,
        )
        cached_enrichment = (
            semantic_cache.get(mode, contract_type, rep) if should_try_llm else None
        )
        if cached_enrichment:
            enrich = external_llm.ExternalEnrichmentResult(
                used=True,
                text=cached_enrichment,
                model="semantic-cache",
                input_tokens=None,
                output_tokens=None,
                cost_usd=0.0,
            )
        else:
            enrich, _ = external_llm.maybe_enrich_opinion(
                mode=mode,
                gate_triggered=gate_on,
                api_key=settings.openai_api_key,
                base_url=str(settings.openai_base_url),
                model=str(settings.openai_model),
                executive_summary=executive_summary,
                main_risks=main_risks,
                recommendations=recommendations,
                contract_type=contract_type,
                risk_level=risk_level,
                document_kind=document_kind,
                excerpt=rep,
            )
            if enrich.used:
                semantic_cache.put(f"{mode}:{contract_type}", mode, contract_type, rep, enrich.text)
        if enrich.used:
            self._last_external_used = True
            self._last_external_provider = "internal" if enrich.model == "semantic-cache" else "openai"
            self._last_external_model = enrich.model
            self._last_external_cost_usd = enrich.cost_usd
            executive_summary = (
                executive_summary
                + "\n\n---\nParecer complementar (IA generativa): "
                + enrich.text
            )
            self._record_step(
                steps,
                "external_llm",
                s_llm,
                provider="internal" if enrich.model == "semantic-cache" else "openai",
                model=enrich.model,
                confidence=0.72,
                input_tokens=enrich.input_tokens,
                output_tokens=enrich.output_tokens,
                estimated_cost_usd=enrich.cost_usd,
            )

        clauses = [
            Clause(
                numero=str(c.get("numero", "")),
                tipo=str(c.get("tipo", "geral")),
                titulo=str(c.get("titulo", "")),
                texto=str(c.get("texto", "")),
            )
            for c in raw_clauses
        ]

        # Timeline
        raw_timeline = self._preprocessor.extract_timeline(
            cleaned, list(entities.datas)
        )
        timeline = [
            TimelineEvent(data=e["data"], evento=e["evento"])
            for e in raw_timeline
        ]

        # Resumo por seção
        section_summary: dict[str, str] = {}
        if sections.get("partes"):
            section_summary["partes"] = "; ".join(sections["partes"][:4])
        if sections.get("clausulas"):
            section_summary["principais_clausulas"] = "; ".join(sections["clausulas"][:3])
        if sections.get("valores"):
            section_summary["valores_mencionados"] = "; ".join(sections["valores"][:6])
        if sections.get("datas"):
            section_summary["datas_relevantes"] = "; ".join(sections["datas"][:4])

        elapsed = time.perf_counter() - t0
        self._last_steps = steps

        return AnalysisResult(
            contract_type=contract_type,
            document_kind=document_kind,
            risk_score=risk_score,
            risk_level=risk_level,
            attention_points=attention_points,
            executive_summary=executive_summary,
            main_risks=main_risks,
            recommendations=recommendations,
            positive_points=positive_points,
            win_prediction=win_prediction,
            win_probability=win_probability,
            win_confidence=win_confidence,
            outcome_probabilities=outcome_probs,
            fee_estimate_min=float(fee_out.get("fee_estimate_min", 0.0)),
            fee_estimate_max=float(fee_out.get("fee_estimate_max", 0.0)),
            fee_estimate_suggested=float(fee_out.get("fee_estimate_suggested", 0.0)),
            entities=entities,
            similar_cases=similar_cases,
            similar_cases_notice=similar_notice if isinstance(similar_notice, str) else None,
            clauses=clauses,
            timeline=timeline,
            section_summary=section_summary,
            urgency=urgency,
            compliance=compliance,
            obligations=obligations,
            processing_time_seconds=round(elapsed, 4),
        )

    def compare(self, text_a: str, text_b: str) -> contract_differ.CompareResult:
        cleaned_a = self._preprocessor.clean(text_a or "")
        cleaned_b = self._preprocessor.clean(text_b or "")
        return contract_differ.compare_texts(
            cleaned_a,
            cleaned_b,
            self._preprocessor,
            str(self._models_dir),
        )
