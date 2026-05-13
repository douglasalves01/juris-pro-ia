"""Contrato HTTP único da API de análise jurídica (sem versionamento de rotas)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


RiskLevel = Literal["BAIXO", "MEDIO", "ALTO", "CRITICO"]
Severity = Literal["low", "medium", "high", "critical"]
Source = Literal["hf", "rules", "llm", "hybrid"]
Provider = Literal["huggingface", "openai", "gemini", "internal", "rules"]


class DocumentBlock(BaseModel):
    type: str
    legalArea: str
    language: str = "pt-BR"
    pageCount: int | None = None
    textQuality: float = Field(ge=0.0, le=1.0)
    summary: str


class RiskBlock(BaseModel):
    score: int = Field(ge=0, le=100)
    level: RiskLevel
    rationale: str
    confidence: float = Field(ge=0.0, le=1.0)


class AttentionPointItem(BaseModel):
    severity: Severity
    clause: str
    title: str | None = None
    description: str
    recommendation: str | None = None
    page: int | None = None
    evidence: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    source: Source


class EntityItem(BaseModel):
    type: str
    value: str
    normalizedValue: str | None = None
    page: int | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    source: Source


class SimilarCaseItem(BaseModel):
    caseId: str | None = None
    tribunal: str
    number: str
    similarity: float
    outcome: str
    summary: str
    relevanceReason: str | None = None


class FeesEstimate(BaseModel):
    min: float | None = None
    suggested: float | None = None
    max: float | None = None
    rationale: str | None = None


class OutcomeProbability(BaseModel):
    value: float = Field(ge=0.0, le=1.0)
    rationale: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)


class UrgencyPayload(BaseModel):
    score: int = Field(ge=0, le=100)
    level: Literal["IMEDIATO", "URGENTE", "NORMAL", "BAIXO"]
    rationale: str


class ComplianceItemPayload(BaseModel):
    id: str
    description: str
    status: Literal["ok", "fail", "warning", "na"]
    evidence: str | None = None


class CompliancePayload(BaseModel):
    regulation: str
    items: list[ComplianceItemPayload]


class ObligationPayload(BaseModel):
    subject: str
    obligation: str
    deadline: str | None = None
    deadlineAbsolute: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)


class FinalOpinion(BaseModel):
    title: str
    executiveSummary: str
    legalAnalysis: str
    mainRisks: list[str] = Field(min_length=1)
    recommendations: list[str] = Field(min_length=1)
    positivePoints: list[str] = Field(min_length=1)
    limitations: list[str]


class AnalysisPayload(BaseModel):
    document: DocumentBlock
    risk: RiskBlock
    attentionPoints: list[AttentionPointItem]
    entities: list[EntityItem]
    similarCases: list[SimilarCaseItem]
    fees: FeesEstimate | None = None
    outcomeProbability: OutcomeProbability
    urgency: UrgencyPayload | None = None
    compliance: list[CompliancePayload] = Field(default_factory=list)
    obligations: list[ObligationPayload] = Field(default_factory=list)
    finalOpinion: FinalOpinion | None = None


class TraceStep(BaseModel):
    step: str
    provider: Provider
    model: str | None = None
    modelVersion: str | None = None
    durationMs: int = Field(ge=0)
    startedAt: str | None = None
    finishedAt: str | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    inputTokens: int | None = None
    outputTokens: int | None = None
    estimatedCostUsd: float | None = None


class TracePayload(BaseModel):
    pipelineVersion: str
    startedAt: str
    finishedAt: str
    durationMs: int
    mode: Literal["fast", "standard", "deep"]
    externalApiUsed: bool
    externalProvider: Literal["openai", "gemini"] | None = None
    externalModel: str | None = None
    estimatedCostUsd: float | None = None
    localModelCostEstimateUsd: float | None = None
    steps: list[TraceStep]


class AnalysisResponse(BaseModel):
    jobId: str
    contractId: str
    status: Literal["done"]
    result: AnalysisPayload
    trace: TracePayload


class AnalysisErrorBody(BaseModel):
    code: Literal[
        "UNSUPPORTED_FILE",
        "TEXT_EXTRACTION_FAILED",
        "OCR_FAILED",
        "MODEL_UNAVAILABLE",
        "EXTERNAL_API_FAILED",
        "OUTPUT_VALIDATION_FAILED",
        "DOCUMENT_TOO_LARGE",
        "TIMEOUT",
        "JOB_ALREADY_ACTIVE",
        "UNKNOWN",
    ]
    message: str
    retryable: bool
    detail: Any | None = None


class AnalysisErrorResponse(BaseModel):
    jobId: str
    contractId: str
    status: Literal["error"]
    error: AnalysisErrorBody
    trace: TracePayload | None = None


# Feature 1.1 — Resumo para cliente leigo
class PlainSummaryRequest(BaseModel):
    jobId: str | None = None
    contractId: str | None = None
    text: str | None = None
    level: Literal["leigo", "intermediario", "tecnico"] = "leigo"
    include_pdf: bool = False

class PlainSummaryResponse(BaseModel):
    jobId: str | None = None
    summaryText: str
    level: str
    generatedAt: str  # ISO 8601 UTC
    pdfBase64: str | None = None

# Feature 1.2 — Argumentos contrários
class CounterArgumentItem(BaseModel):
    text: str
    strength: Literal["forte", "medio", "fraco"]
    category: str

class CounterArgumentsRequest(BaseModel):
    text: str | None = None
    jobId: str | None = None
    contractId: str | None = None
    maxArguments: int = Field(default=5, ge=1, le=20)

class CounterArgumentsResponse(BaseModel):
    jobId: str | None = None
    arguments: list[CounterArgumentItem]
    generatedAt: str  # ISO 8601 UTC

# Feature 1.3 — Urgência processual
class UrgencyRequest(BaseModel):
    text: str

class UrgencyResponse(BaseModel):
    score: int = Field(ge=0, le=100)
    level: Literal["IMEDIATO", "URGENTE", "NORMAL", "BAIXO"]
    rationale: str
    generatedAt: str  # ISO 8601 UTC

class ObligationsRequest(BaseModel):
    text: str

class ObligationsResponse(BaseModel):
    obligations: list[ObligationPayload]
    generatedAt: str

class QualitySuggestionPayload(BaseModel):
    dimension: Literal["completeness", "coherence", "citations", "language"]
    issue: str
    fix: str

class QualityRequest(BaseModel):
    text: str

class QualityResponse(BaseModel):
    score: int = Field(ge=0, le=100)
    dimensions: dict[str, int]
    suggestions: list[QualitySuggestionPayload]

class MonitorSubscribeRequest(BaseModel):
    caseId: str
    contractId: str | None = None
    threshold: float = Field(default=0.85, ge=0.0, le=1.0)
    webhookUrl: str | None = None

class MonitorSubscribeResponse(BaseModel):
    caseId: str
    contractId: str | None = None
    threshold: float
    subscribedAt: str

class MonitorAlertItem(BaseModel):
    caseId: str
    newDecisionId: str
    similarity: float = Field(ge=0.0, le=1.0)
    tribunal: str
    summary: str
    url: str | None = None
    createdAt: str

class MonitorAlertsResponse(BaseModel):
    caseId: str
    alerts: list[MonitorAlertItem]
    collectedAt: str

class DraftContext(BaseModel):
    parties: Any | None = None
    subject: Any | None = None
    facts: Any | None = None
    claims: Any | None = None

class DraftRequest(BaseModel):
    documentType: Literal["peticao_inicial", "contestacao", "contrato"]
    context: DraftContext
    style: Literal["formal", "conciso"] = "formal"
    firmId: str | None = None

class DraftSection(BaseModel):
    title: str
    content: str
    needsReview: bool
    confidence: float = Field(ge=0.0, le=1.0)

class DraftResponse(BaseModel):
    draft: str
    sections: list[DraftSection]
    disclaimer: str

class FirmKnowledgeDocument(BaseModel):
    documentId: str | None = None
    type: Literal["peca_aprovada", "parecer", "modelo_interno", "outro"] = "outro"
    title: str
    text: str
    sourceUrl: str | None = None

class FirmKnowledgeIngestRequest(BaseModel):
    documents: list[FirmKnowledgeDocument] = Field(min_length=1, max_length=50)

class FirmKnowledgeIngestResponse(BaseModel):
    jobId: str
    firmId: str
    status: Literal["queued", "indexed"]
    documentsReceived: int

class FirmKnowledgeStatsResponse(BaseModel):
    firmId: str
    collection: str
    documentsIndexed: int
    lastUpdatedAt: str | None = None

# Feature 1.4 — Métricas de latência
class StepMetric(BaseModel):
    step: str
    avgDurationMs: float
    p95DurationMs: int
    callCount: int

class SemanticCacheMetrics(BaseModel):
    cacheHits: int
    cacheMisses: int
    cacheHitRate: float = Field(ge=0.0, le=1.0)
    cacheEntries: int

class PipelineMetricsResponse(BaseModel):
    steps: list[StepMetric]
    collectedAt: str  # ISO 8601 UTC
    semanticCache: SemanticCacheMetrics | None = None
