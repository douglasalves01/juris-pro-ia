# Design Técnico — Sprint 1 do JurisPro IA

## Visão Geral

Este documento descreve o design técnico das quatro funcionalidades da Sprint 1 do JurisPro IA. Todas as features são incrementais ao microserviço FastAPI existente e não alteram o pipeline principal de forma destrutiva.

**Stack:** Python 3.11+, FastAPI, Pydantic v2, fpdf2, pytest, httpx.

**Princípios de design:**
- Sem autenticação nos endpoints públicos de análise (padrão do projeto).
- Fallback determinístico quando LLM externo não estiver configurado.
- Sem dependências novas além das já presentes em `requirements.txt`.
- Thread-safety para estruturas de estado compartilhado (buffer de métricas).

---

## Arquitetura

```
api/
├── main.py                          ← novos endpoints: /analyze/summary/plain,
│                                      /analyze/counter-arguments, /analyze/urgency,
│                                      GET /metrics/pipeline
├── schemas/
│   └── analysis.py                  ← TraceStep estendido (startedAt, finishedAt)
│                                      + novos schemas: PlainSummaryResponse,
│                                        CounterArgumentsResponse, UrgencyResponse,
│                                        PipelineMetricsResponse
├── ml/
│   ├── pipeline.py                  ← _record_step estendido, UrgencyBlock no
│   │                                  AnalysisResult, buffer circular de métricas
│   └── models/
│       └── urgency_classifier.py    ← novo módulo de regras heurísticas
└── services/
    └── plain_summary_service.py     ← lógica de reescrita (LLM + fallback de regras)
    └── counter_arguments_service.py ← lógica de contra-argumentos (LLM + fallback)

tests/
├── test_plain_summary.py
├── test_counter_arguments.py
└── test_pipeline_metrics.py

api/ml/tests/
└── test_urgency_classifier.py
```

---

## Componentes e Interfaces

### 1.1 — Serviço de Resumo para Cliente Leigo

**Arquivo:** `api/services/plain_summary_service.py`

```python
def simplify_by_rules(text: str, level: str) -> str:
    """Fallback: remove jargão, encurta frases, substitui termos técnicos."""
    ...

def rewrite_with_llm(
    text: str,
    level: str,
    api_key: str,
    base_url: str,
    model: str,
) -> str:
    """Chama LLM externo via httpx. Retorna texto reescrito ou levanta exceção."""
    ...

def generate_summary(
    executive_summary: str,
    level: str,
    api_key: str | None,
    base_url: str,
    model: str,
) -> str:
    """Orquestra: tenta LLM, cai no fallback em caso de falha ou ausência de chave."""
    ...

def generate_pdf_base64(summary_text: str) -> str:
    """Gera PDF com fpdf2 e retorna string base64."""
    ...
```

**Endpoint:** `POST /analyze/summary/plain`

```python
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
```

**Lógica de resolução do texto de entrada:**
1. Se `text` fornecido → usar diretamente.
2. Se `jobId` fornecido → buscar `executive_summary` do job em `app.state.analysis_jobs`.
3. Se `contractId` fornecido → buscar job mais recente com esse `contractId` em `app.state.analysis_jobs`.
4. Se nenhum → HTTP 422.

**Simplificação por regras (fallback):**
- Substituir termos jurídicos por equivalentes comuns (dicionário de ~30 termos: "exequente" → "quem cobra", "réu" → "parte acusada", etc.).
- Dividir frases com mais de 40 palavras em frases menores usando pontuação.
- Para `level = "tecnico"`: retornar o texto original sem modificação.
- Para `level = "intermediario"`: aplicar apenas substituição de termos mais obscuros.
- Para `level = "leigo"`: aplicar substituição completa e encurtamento de frases.

**Prompt LLM (quando disponível):**
```
Sistema: Você é um assistente jurídico que explica documentos legais em linguagem {level}.
Usuário: Reescreva o seguinte resumo jurídico em linguagem {descricao_nivel}. 
Mantenha os fatos, mas use palavras simples. Máximo 200 palavras.
Texto: {executive_summary}
```

---

### 1.2 — Serviço de Argumentos Contrários

**Arquivo:** `api/services/counter_arguments_service.py`

```python
class CounterArgument(TypedDict):
    text: str
    strength: Literal["forte", "medio", "fraco"]
    category: str

def build_from_attention_points(
    attention_points: list[dict],
    max_arguments: int,
) -> list[CounterArgument]:
    """Fallback: reformula attentionPoints como argumentos adversariais."""
    ...

def build_with_llm(
    text: str,
    max_arguments: int,
    api_key: str,
    base_url: str,
    model: str,
) -> list[CounterArgument]:
    """Chama LLM com prompt adversarial. Retorna lista ou levanta exceção."""
    ...

def generate_counter_arguments(
    text: str,
    attention_points: list[dict],
    max_arguments: int,
    api_key: str | None,
    base_url: str,
    model: str,
) -> list[CounterArgument]:
    """Orquestra: tenta LLM, cai no fallback."""
    ...
```

**Endpoint:** `POST /analyze/counter-arguments`

```python
class CounterArgumentsRequest(BaseModel):
    text: str | None = None
    jobId: str | None = None
    contractId: str | None = None
    maxArguments: int = Field(default=5, ge=1, le=20)

class CounterArgumentItem(BaseModel):
    text: str
    strength: Literal["forte", "medio", "fraco"]
    category: str

class CounterArgumentsResponse(BaseModel):
    jobId: str | None = None
    arguments: list[CounterArgumentItem]
    generatedAt: str  # ISO 8601 UTC
```

**Mapeamento de severidade → força:**
- `"alta"` / `"crítica"` → `"forte"`
- `"média"` → `"medio"`
- `"baixa"` → `"fraco"`

**Mapeamento de tipo → categoria:**
- `"lgpd_compliance"` → `"Proteção de Dados"`
- `"penalty_clause"` → `"Cláusula Penal"`
- `"termination_clause"` → `"Rescisão"`
- `"jurisdiction_clause"` → `"Competência"`
- `"liability_limitation"` → `"Responsabilidade"`
- `"intellectual_property"` → `"Propriedade Intelectual"`
- `"dispute_resolution"` → `"Resolução de Conflitos"`
- outros → `"Geral"`

**Prompt LLM adversarial:**
```
Sistema: Você é um advogado adversário experiente. Identifique os pontos mais fracos do documento.
Usuário: Analise o texto jurídico abaixo e liste {max_arguments} argumentos que a parte contrária 
poderia usar. Para cada argumento, indique: texto do argumento, força ("forte"/"medio"/"fraco") 
e categoria jurídica. Responda em JSON.
Texto: {text[:3000]}
```

---

### 1.3 — Classificador de Urgência Processual

**Arquivo:** `api/ml/models/urgency_classifier.py`

```python
from dataclasses import dataclass
from datetime import date, datetime, timezone
import re

@dataclass
class UrgencyResult:
    score: int          # 0–100
    level: str          # "IMEDIATO" | "URGENTE" | "NORMAL" | "BAIXO"
    rationale: str

_KEYWORDS_IMEDIATO = frozenset([
    "liminar", "tutela de urgência", "tutela antecipada", "prazo fatal",
    "audiência", "penhora", "arresto", "sequestro", "busca e apreensão",
])

_KEYWORDS_URGENTE = frozenset([
    "prazo", "intimação", "citação", "recurso", "apelação", "embargos",
    "contestação", "impugnação", "notificação",
])

def _parse_date(date_str: str) -> date | None:
    """Tenta parsear string de data em vários formatos. Retorna None se falhar."""
    ...

def _days_until(d: date) -> int:
    """Retorna número de dias entre hoje e a data fornecida (negativo se passada)."""
    ...

def classify(
    text: str,
    entities_datas: list[str],
    contract_type: str = "",
) -> UrgencyResult:
    """
    Classifica urgência por regras heurísticas.
    Prioridade: datas próximas > palavras-chave imediatas > palavras-chave urgentes > padrão.
    """
    ...
```

**Algoritmo de classificação (ordem de prioridade):**

1. **Datas próximas (máxima prioridade):**
   - Parsear cada string em `entities_datas` tentando formatos: `DD/MM/YYYY`, `DD-MM-YYYY`, `DD de <mês> de YYYY`, `MM/YYYY`.
   - Se alguma data estiver entre hoje e +7 dias → `IMEDIATO`, score = 90.
   - Se alguma data estiver entre +8 e +30 dias → `URGENTE`, score = 65.

2. **Palavras-chave no texto (segunda prioridade):**
   - Busca case-insensitive com `re.search`.
   - Se alguma keyword de `_KEYWORDS_IMEDIATO` encontrada → `IMEDIATO`, score = 85.
   - Se alguma keyword de `_KEYWORDS_URGENTE` encontrada → `URGENTE`, score = 60.

3. **Texto muito curto (< 50 palavras) ou sem indicadores:**
   - `BAIXO`, score = 10.

4. **Padrão:**
   - `NORMAL`, score = 30.

**Integração no Pipeline (`api/ml/pipeline.py`):**

```python
# Após o step risk_scoring, antes de validate_output:
s0 = time.perf_counter()
from api.ml.models.urgency_classifier import classify as classify_urgency
urgency_out = classify_urgency(
    text=cleaned,
    entities_datas=list(entities.datas),
    contract_type=contract_type,
)
self._record_step(steps, "urgency_classification", s0, model="rules", confidence=0.9)
```

**Novo campo em `AnalysisResult`:**
```python
class UrgencyBlock(BaseModel):
    score: int = Field(ge=0, le=100)
    level: str  # "IMEDIATO" | "URGENTE" | "NORMAL" | "BAIXO"
    rationale: str

class AnalysisResult(BaseModel):
    ...
    urgency: UrgencyBlock = Field(
        default_factory=lambda: UrgencyBlock(score=0, level="BAIXO", rationale="Não classificado.")
    )
```

**Endpoint dedicado:** `POST /analyze/urgency`

```python
class UrgencyRequest(BaseModel):
    text: str

class UrgencyResponse(BaseModel):
    score: int
    level: str
    rationale: str
    generatedAt: str  # ISO 8601 UTC
```

O endpoint extrai entidades de data do texto usando `TextPreprocessor.extract_sections` e chama `classify_urgency` diretamente, sem executar o pipeline completo.

---

### 1.4 — Latência por Step no Trace

**Modificações em `api/ml/pipeline.py`:**

```python
def _record_step(
    self,
    steps: list[dict[str, object]],
    step: str,
    started: float,          # time.perf_counter() antes do step
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
    duration_ms = max(0, int((now - started) * 1000))  # nunca None
    started_at_iso = datetime.fromtimestamp(
        time.time() - (now - started), tz=timezone.utc
    ).isoformat()
    finished_at_iso = datetime.now(timezone.utc).isoformat()
    steps.append({
        "step": step,
        "provider": provider,
        "model": model,
        "modelVersion": model_version,
        "durationMs": duration_ms,          # sempre int >= 0
        "startedAt": started_at_iso,        # novo campo
        "finishedAt": finished_at_iso,      # novo campo
        "confidence": confidence,
        "inputTokens": input_tokens,
        "outputTokens": output_tokens,
        "estimatedCostUsd": estimated_cost_usd,
    })
```

**Buffer circular de métricas:**

```python
# Em api/main.py (ou api/metrics.py separado)
import collections
import threading
import statistics

_METRICS_BUFFER_SIZE = 100
_metrics_buffer: collections.deque[list[dict]] = collections.deque(maxlen=_METRICS_BUFFER_SIZE)
_metrics_lock = threading.Lock()

def record_pipeline_metrics(steps: list[dict]) -> None:
    """Adiciona steps de uma execução ao buffer circular. Thread-safe."""
    with _metrics_lock:
        _metrics_buffer.append(steps)

def compute_pipeline_metrics() -> dict:
    """Calcula avg e p95 por step a partir do buffer."""
    with _metrics_lock:
        snapshot = list(_metrics_buffer)
    
    per_step: dict[str, list[int]] = {}
    for execution_steps in snapshot:
        for step in execution_steps:
            name = step.get("step", "unknown")
            dur = step.get("durationMs")
            if isinstance(dur, int) and dur >= 0:
                per_step.setdefault(name, []).append(dur)
    
    result = []
    for step_name, durations in per_step.items():
        avg = sum(durations) / len(durations)
        sorted_d = sorted(durations)
        p95_idx = max(0, int(len(sorted_d) * 0.95) - 1)
        p95 = sorted_d[p95_idx]
        result.append({
            "step": step_name,
            "avgDurationMs": round(avg, 2),
            "p95DurationMs": p95,
            "callCount": len(durations),
        })
    
    return result
```

**Modificações em `api/schemas/analysis.py`:**

```python
class TraceStep(BaseModel):
    step: str
    provider: Provider
    model: str | None = None
    modelVersion: str | None = None
    durationMs: int = Field(ge=0)          # nunca None — alterado de int | None
    startedAt: str | None = None           # novo campo
    finishedAt: str | None = None          # novo campo
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    inputTokens: int | None = None
    outputTokens: int | None = None
    estimatedCostUsd: float | None = None
```

**Endpoint:** `GET /metrics/pipeline`

```python
class StepMetric(BaseModel):
    step: str
    avgDurationMs: float
    p95DurationMs: int
    callCount: int

class PipelineMetricsResponse(BaseModel):
    steps: list[StepMetric]
    collectedAt: str  # ISO 8601 UTC
```

---

## Modelos de Dados

### Novos Schemas em `api/schemas/analysis.py`

```python
# Feature 1.1
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
    generatedAt: str
    pdfBase64: str | None = None

# Feature 1.2
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
    generatedAt: str

# Feature 1.3
class UrgencyRequest(BaseModel):
    text: str

class UrgencyResponse(BaseModel):
    score: int = Field(ge=0, le=100)
    level: Literal["IMEDIATO", "URGENTE", "NORMAL", "BAIXO"]
    rationale: str
    generatedAt: str

# Feature 1.4
class StepMetric(BaseModel):
    step: str
    avgDurationMs: float
    p95DurationMs: int
    callCount: int

class PipelineMetricsResponse(BaseModel):
    steps: list[StepMetric]
    collectedAt: str
```

### Extensão de `AnalysisResult` em `api/ml/pipeline.py`

```python
class UrgencyBlock(BaseModel):
    score: int = Field(ge=0, le=100)
    level: Literal["IMEDIATO", "URGENTE", "NORMAL", "BAIXO"]
    rationale: str

class AnalysisResult(BaseModel):
    # ... campos existentes ...
    urgency: UrgencyBlock = Field(
        default_factory=lambda: UrgencyBlock(
            score=0, level="BAIXO", rationale="Não classificado."
        )
    )
```

---

## Propriedades de Correção

*Uma propriedade é uma característica ou comportamento que deve ser verdadeiro em todas as execuções válidas do sistema — essencialmente, uma declaração formal sobre o que o sistema deve fazer. As propriedades servem como ponte entre especificações legíveis por humanos e garantias de correção verificáveis por máquina.*

### Propriedade 1: Qualquer entrada válida produz resposta com schema correto (Resumo Leigo)

*Para qualquer* combinação válida de `(text, jobId, contractId, level, include_pdf)` onde ao menos um dos três primeiros campos está presente, a resposta do endpoint `/analyze/summary/plain` deve conter os campos `summaryText` (string não vazia), `level` (igual ao solicitado), `generatedAt` (string ISO 8601) e `pdfBase64` (string ou null conforme `include_pdf`).

**Valida: Requisitos 1.2, 1.10**

---

### Propriedade 2: Valores inválidos de `level` sempre retornam 422

*Para qualquer* string que não seja `"leigo"`, `"intermediario"` ou `"tecnico"`, o endpoint `/analyze/summary/plain` deve retornar HTTP 422.

**Valida: Requisito 1.5**

---

### Propriedade 3: Fallback de regras sempre produz texto não vazio

*Para qualquer* `executive_summary` não vazio e qualquer `level` válido, a função `simplify_by_rules` deve retornar uma string não vazia.

**Valida: Requisito 1.7**

---

### Propriedade 4: PDF gerado é base64 decodificável para bytes de PDF válido

*Para qualquer* `summaryText` não vazio, a função `generate_pdf_base64` deve retornar uma string que, quando decodificada de base64, começa com os bytes mágicos `%PDF`.

**Valida: Requisito 1.8**

---

### Propriedade 5: Qualquer entrada válida produz resposta com schema correto (Contra-argumentos)

*Para qualquer* combinação válida de `(text, jobId, contractId, maxArguments)` onde ao menos um dos três primeiros campos está presente, a resposta do endpoint `/analyze/counter-arguments` deve conter `arguments` (lista), `generatedAt` (string ISO 8601) e cada item da lista deve ter `text`, `strength` e `category`.

**Valida: Requisitos 2.2, 2.8, 2.10**

---

### Propriedade 6: `maxArguments` é respeitado como limite superior

*Para qualquer* `maxArguments` válido (1–20) e qualquer texto de entrada, `len(response.arguments) <= maxArguments`.

**Valida: Requisito 2.9**

---

### Propriedade 7: Fallback de contra-argumentos preserva campos obrigatórios

*Para qualquer* lista de `attentionPoints` (incluindo lista vazia), a função `build_from_attention_points` deve retornar uma lista onde cada item tem `text` (string não vazia), `strength` (um de `"forte"`, `"medio"`, `"fraco"`) e `category` (string não vazia).

**Valida: Requisitos 2.7, 2.8**

---

### Propriedade 8: Palavras-chave de urgência imediata sempre produzem IMEDIATO

*Para qualquer* texto que contenha ao menos uma das palavras-chave de urgência imediata (case-insensitive), `classify()` deve retornar `level = "IMEDIATO"` e `score >= 80`.

**Valida: Requisito 3.2**

---

### Propriedade 9: Datas próximas determinam nível de urgência correto

*Para qualquer* lista `entities_datas` contendo ao menos uma data dentro de 7 dias corridos, `classify()` deve retornar `level = "IMEDIATO"` e `score >= 80`. *Para qualquer* lista contendo ao menos uma data entre 8 e 30 dias (e nenhuma dentro de 7), deve retornar `level = "URGENTE"`.

**Valida: Requisitos 3.4, 3.5**

---

### Propriedade 10: `rationale` nunca é vazio

*Para qualquer* entrada válida em `classify()`, o campo `rationale` do resultado deve ser uma string com ao menos 10 caracteres.

**Valida: Requisito 3.8**

---

### Propriedade 11: Todos os steps do pipeline têm `durationMs` não negativo

*Para qualquer* execução do pipeline com qualquer texto de entrada, todos os steps em `pipeline.last_steps` devem ter `durationMs` como inteiro `>= 0` (nunca `None`).

**Valida: Requisito 4.1**

---

### Propriedade 12: Todos os steps têm `startedAt` e `finishedAt` preenchidos

*Para qualquer* execução do pipeline, todos os steps em `pipeline.last_steps` devem ter `startedAt` e `finishedAt` como strings não vazias no formato ISO 8601.

**Valida: Requisito 4.3**

---

### Propriedade 13: Cálculo de `avgDurationMs` e `p95DurationMs` é matematicamente correto

*Para qualquer* conjunto de durações registradas no buffer para um dado step, `avgDurationMs` deve ser igual à média aritmética das durações e `p95DurationMs` deve ser o valor no índice `floor(n * 0.95)` da lista ordenada.

**Valida: Requisitos 4.6, 4.9**

---

### Propriedade 14: Buffer circular não excede 100 entradas

*Para qualquer* número de execuções registradas (incluindo > 100), o buffer deve conter no máximo 100 entradas.

**Valida: Requisito 4.4**

---

## Tratamento de Erros

### Feature 1.1 — Resumo para Cliente Leigo

| Situação | HTTP | Código de erro |
|---|---|---|
| Nenhum de `text`, `jobId`, `contractId` fornecido | 422 | `MISSING_INPUT` |
| `level` inválido | 422 | (validação Pydantic) |
| `jobId` não encontrado em `analysis_jobs` | 404 | `JOB_NOT_FOUND` |
| Falha na geração do PDF | 500 | `PDF_GENERATION_FAILED` |
| Falha no LLM externo | 200 | (fallback silencioso) |

### Feature 1.2 — Argumentos Contrários

| Situação | HTTP | Código de erro |
|---|---|---|
| Nenhum de `text`, `jobId`, `contractId` fornecido | 422 | `MISSING_INPUT` |
| `maxArguments` fora do intervalo [1, 20] | 422 | (validação Pydantic) |
| `jobId` não encontrado | 404 | `JOB_NOT_FOUND` |
| Falha no LLM externo | 200 | (fallback silencioso) |

### Feature 1.3 — Urgência Processual

| Situação | HTTP | Código de erro |
|---|---|---|
| `text` vazio ou ausente no endpoint `/analyze/urgency` | 422 | (validação Pydantic) |

### Feature 1.4 — Métricas de Latência

Sem erros esperados. O endpoint `/metrics/pipeline` sempre retorna 200 (lista vazia se buffer vazio).

---

## Estratégia de Testes

### Abordagem Dual

- **Testes de unidade/exemplo:** verificam comportamentos específicos, casos de borda e integração entre componentes.
- **Testes de propriedade (PBT):** verificam propriedades universais usando `hypothesis` (já disponível como dependência transitiva via `pytest`).

### Biblioteca de PBT

Usar `hypothesis` com estratégias customizadas para gerar textos jurídicos sintéticos, listas de datas e listas de `attentionPoints`.

Cada teste de propriedade deve rodar com mínimo de 100 iterações (`@settings(max_examples=100)`).

Tag de referência: `# Feature: sprint-1, Property N: <texto da propriedade>`

### Arquivos de Teste

| Arquivo | Cobertura |
|---|---|
| `tests/test_plain_summary.py` | Endpoint `/analyze/summary/plain`, serviço de resumo, geração de PDF |
| `tests/test_counter_arguments.py` | Endpoint `/analyze/counter-arguments`, serviço de contra-argumentos |
| `api/ml/tests/test_urgency_classifier.py` | Módulo `urgency_classifier.py` (unidade) |
| `tests/test_pipeline_metrics.py` | Endpoint `/metrics/pipeline`, buffer circular, cálculos estatísticos |

### Padrão de Fixture

Seguir o padrão de `tests/test_analyze_http_integration.py`:
- `TestClient` do FastAPI com `app.state.pipeline` mockado via `MagicMock`.
- `AnalysisPipeline._instance = None` no topo do arquivo para isolar o singleton.
- Fixture `client` que injeta o mock no `app.state`.

### Testes de Propriedade com `hypothesis`

```python
from hypothesis import given, settings
from hypothesis import strategies as st

# Propriedade 3: fallback de regras sempre produz texto não vazio
@given(
    text=st.text(min_size=1, max_size=2000),
    level=st.sampled_from(["leigo", "intermediario", "tecnico"]),
)
@settings(max_examples=100)
def test_simplify_by_rules_never_empty(text, level):
    # Feature: sprint-1, Property 3: fallback de regras sempre produz texto não vazio
    result = simplify_by_rules(text, level)
    assert isinstance(result, str)
    assert len(result) > 0
```

### Mocks Necessários

- `api.ml.external_llm.maybe_enrich_opinion` → para isolar chamadas ao LLM.
- `api.services.plain_summary_service.rewrite_with_llm` → para testar fallback.
- `api.services.counter_arguments_service.build_with_llm` → para testar fallback.
- `fpdf2.FPDF` → para testar falha na geração de PDF.
