# Plano de Implementação — Sprint 1 do JurisPro IA

## Visão Geral

Implementação incremental das quatro features da Sprint 1. Cada tarefa produz código funcional e integrado; nenhum código fica órfão. A ordem respeita dependências: schemas → serviços → endpoints → testes.

---

## Tarefas

- [x] 1. Estender schemas Pydantic com os novos tipos da Sprint 1
  - Adicionar `UrgencyBlock` ao `api/ml/pipeline.py` (antes de `AnalysisResult`).
  - Adicionar campo `urgency: UrgencyBlock` ao `AnalysisResult` com `default_factory`.
  - Alterar `TraceStep` em `api/schemas/analysis.py`: `durationMs: int = Field(ge=0)` (remover `| None`), adicionar `startedAt: str | None = None` e `finishedAt: str | None = None`.
  - Adicionar ao `api/schemas/analysis.py`: `PlainSummaryRequest`, `PlainSummaryResponse`, `CounterArgumentItem`, `CounterArgumentsRequest`, `CounterArgumentsResponse`, `UrgencyRequest`, `UrgencyResponse`, `StepMetric`, `PipelineMetricsResponse`.
  - _Requisitos: 1.10, 2.8, 2.10, 3.9, 4.1, 4.2, 4.11_

- [x] 2. Corrigir `_record_step` no pipeline para garantir `durationMs` sempre preenchido e adicionar `startedAt`/`finishedAt`
  - [x] 2.1 Modificar `_record_step` em `api/ml/pipeline.py`:
    - Calcular `duration_ms = max(0, int((time.perf_counter() - started) * 1000))`.
    - Calcular `started_at_iso` e `finished_at_iso` como strings ISO 8601 UTC.
    - Incluir `"startedAt"` e `"finishedAt"` no dict appendado a `steps`.
    - Garantir que `"durationMs"` nunca seja `None`.
    - _Requisitos: 4.1, 4.3_
  - [x]* 2.2 Escrever teste de propriedade para `durationMs` nunca nulo
    - **Propriedade 11: Todos os steps do pipeline têm `durationMs` não negativo**
    - Usar `hypothesis` para gerar textos variados e verificar que todos os steps têm `durationMs >= 0`.
    - **Valida: Requisito 4.1**
  - [x]* 2.3 Escrever teste de propriedade para `startedAt`/`finishedAt` sempre preenchidos
    - **Propriedade 12: Todos os steps têm `startedAt` e `finishedAt` preenchidos**
    - Verificar que todos os steps têm strings ISO 8601 não vazias.
    - **Valida: Requisito 4.3**

- [x] 3. Implementar buffer circular de métricas de latência
  - [x] 3.1 Criar estrutura de buffer em `api/main.py`:
    - `_metrics_buffer: collections.deque` com `maxlen=100`.
    - `_metrics_lock: threading.Lock`.
    - Função `record_pipeline_metrics(steps: list[dict]) -> None` (thread-safe).
    - Função `compute_pipeline_metrics() -> list[dict]` que calcula `avgDurationMs`, `p95DurationMs` e `callCount` por step.
    - _Requisitos: 4.4, 4.6, 4.9, 4.10_
  - [x] 3.2 Chamar `record_pipeline_metrics` ao final de cada execução bem-sucedida do pipeline em `_build_analysis_response` (ou no ponto de coleta dos `last_steps`).
    - _Requisitos: 4.4, 4.10_
  - [x]* 3.3 Escrever testes de propriedade para o buffer e cálculos estatísticos em `tests/test_pipeline_metrics.py`:
    - **Propriedade 13: Cálculo de `avgDurationMs` e `p95DurationMs` é matematicamente correto**
    - **Propriedade 14: Buffer circular não excede 100 entradas**
    - Usar `hypothesis` para gerar listas de durações e verificar cálculos.
    - **Valida: Requisitos 4.4, 4.6, 4.9**

- [x] 4. Implementar endpoint `GET /metrics/pipeline`
  - Adicionar rota `GET /metrics/pipeline` em `api/main.py`.
  - Chamar `compute_pipeline_metrics()` e retornar `PipelineMetricsResponse`.
  - Retornar lista vazia quando buffer estiver vazio.
  - _Requisitos: 4.5, 4.6, 4.7, 4.8, 4.11_
  - [x]* 4.1 Escrever testes de exemplo em `tests/test_pipeline_metrics.py`:
    - Verificar que endpoint retorna 200 com lista vazia quando buffer vazio.
    - Verificar schema da resposta com `PipelineMetricsResponse.model_validate`.
    - Verificar que `collectedAt` está presente.
    - _Requisitos: 4.5, 4.7, 4.8_

- [x] 5. Checkpoint — Garantir que todos os testes existentes passam
  - Executar `pytest tests/ api/ml/tests/ -x --tb=short` e corrigir regressões causadas pelas mudanças em `TraceStep` e `_record_step`.
  - Verificar que `durationMs` nunca é `None` nos testes de integração existentes.
  - Garantir que todos os testes passam, perguntar ao usuário se houver dúvidas.

- [x] 6. Implementar módulo `api/ml/models/urgency_classifier.py`
  - [x] 6.1 Criar `api/ml/models/urgency_classifier.py` com:
    - `UrgencyResult` dataclass com `score`, `level`, `rationale`.
    - Constantes `_KEYWORDS_IMEDIATO` e `_KEYWORDS_URGENTE`.
    - Função `_parse_date(date_str: str) -> date | None` com suporte a formatos `DD/MM/YYYY`, `DD-MM-YYYY`, `DD de <mês> de YYYY`, `MM/YYYY`.
    - Função `_days_until(d: date) -> int`.
    - Função `classify(text, entities_datas, contract_type) -> UrgencyResult` com lógica de prioridade: datas próximas > keywords imediatas > keywords urgentes > padrão.
    - _Requisitos: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8_
  - [x]* 6.2 Escrever testes de propriedade em `api/ml/tests/test_urgency_classifier.py`:
    - **Propriedade 8: Palavras-chave de urgência imediata sempre produzem IMEDIATO**
    - Usar `hypothesis` para gerar textos com palavras-chave inseridas em posições aleatórias.
    - **Valida: Requisito 3.2**
  - [x]* 6.3 Escrever testes de propriedade para datas próximas:
    - **Propriedade 9: Datas próximas determinam nível de urgência correto**
    - Gerar datas aleatórias dentro e fora dos intervalos de 7 e 30 dias.
    - **Valida: Requisitos 3.4, 3.5**
  - [x]* 6.4 Escrever testes de propriedade para `rationale` nunca vazio:
    - **Propriedade 10: `rationale` nunca é vazio**
    - Para qualquer entrada válida, `rationale` deve ter ao menos 10 caracteres.
    - **Valida: Requisito 3.8**
  - [x]* 6.5 Escrever testes de exemplo para casos de borda:
    - Texto muito curto → `BAIXO`.
    - Texto vazio → `BAIXO`.
    - Combinação de data próxima + keyword imediata → `IMEDIATO`.
    - _Requisitos: 3.6, 3.7_

- [x] 7. Integrar `urgency_classifier` ao pipeline e ao `AnalysisResult`
  - [x] 7.1 Importar e chamar `classify_urgency` em `api/ml/pipeline.py` após o step `risk_scoring`:
    - Passar `cleaned`, `list(entities.datas)` e `contract_type`.
    - Registrar step `"urgency_classification"` com `_record_step`.
    - Atribuir resultado ao campo `urgency` do `AnalysisResult`.
    - _Requisitos: 3.9, 3.12_
  - [x]* 7.2 Escrever teste de propriedade para `urgency` no `AnalysisResult`:
    - **Propriedade 11 (extensão): Pipeline sempre inclui campo `urgency` com schema válido**
    - Verificar que `result.urgency.score` está em [0, 100] e `result.urgency.level` é um dos quatro valores válidos.
    - **Valida: Requisito 3.9**

- [x] 8. Implementar endpoint `POST /analyze/urgency`
  - Adicionar rota `POST /analyze/urgency` em `api/main.py`.
  - Usar `TextPreprocessor.extract_sections` para extrair datas do texto.
  - Chamar `classify_urgency` diretamente (sem pipeline completo).
  - Retornar `UrgencyResponse`.
  - _Requisitos: 3.10, 3.11_
  - [x]* 8.1 Escrever testes de exemplo em `api/ml/tests/test_urgency_classifier.py` (ou `tests/test_urgency_endpoint.py`):
    - Verificar que endpoint retorna 200 com schema correto para texto válido.
    - Verificar que endpoint retorna 422 para texto vazio.
    - _Requisitos: 3.10, 3.11_

- [x] 9. Implementar serviço de resumo para cliente leigo
  - [x] 9.1 Criar `api/services/plain_summary_service.py` com:
    - Dicionário de substituição de jargão jurídico (~30 termos).
    - Função `simplify_by_rules(text: str, level: str) -> str`.
    - Função `rewrite_with_llm(text, level, api_key, base_url, model) -> str` usando `httpx.Client`.
    - Função `generate_summary(executive_summary, level, api_key, base_url, model) -> str` que orquestra LLM + fallback.
    - Função `generate_pdf_base64(summary_text: str) -> str` usando `fpdf2`.
    - _Requisitos: 1.6, 1.7, 1.8, 1.12_
  - [x]* 9.2 Escrever testes de propriedade em `tests/test_plain_summary.py`:
    - **Propriedade 3: Fallback de regras sempre produz texto não vazio**
    - Usar `hypothesis` para gerar textos variados.
    - **Valida: Requisito 1.7**
  - [x]* 9.3 Escrever testes de propriedade para PDF:
    - **Propriedade 4: PDF gerado é base64 decodificável para bytes de PDF válido**
    - Para qualquer `summaryText` não vazio, verificar que `base64.b64decode(result)` começa com `b"%PDF"`.
    - **Valida: Requisito 1.8**
  - [x]* 9.4 Escrever testes de exemplo para casos específicos:
    - Fallback com `level="tecnico"` retorna texto original sem modificação.
    - LLM falha → fallback é aplicado silenciosamente.
    - `generate_pdf_base64` com texto vazio → levanta exceção.
    - _Requisitos: 1.7, 1.12_

- [x] 10. Implementar endpoint `POST /analyze/summary/plain`
  - Adicionar rota `POST /analyze/summary/plain` em `api/main.py`.
  - Implementar lógica de resolução de texto: `text` > `jobId` > `contractId`.
  - Chamar `generate_summary` do serviço.
  - Chamar `generate_pdf_base64` quando `include_pdf=True`.
  - Tratar erros: 422 para entrada ausente, 404 para `jobId` não encontrado, 500 para falha no PDF.
  - _Requisitos: 1.1, 1.2, 1.3, 1.4, 1.9, 1.10, 1.11_
  - [x]* 10.1 Escrever testes de propriedade em `tests/test_plain_summary.py`:
    - **Propriedade 1: Qualquer entrada válida produz resposta com schema correto**
    - Usar `hypothesis` para gerar combinações de `(text, level, include_pdf)`.
    - **Valida: Requisitos 1.2, 1.10**
  - [x]* 10.2 Escrever testes de propriedade para validação de `level`:
    - **Propriedade 2: Valores inválidos de `level` sempre retornam 422**
    - Usar `hypothesis` para gerar strings que não sejam os três valores válidos.
    - **Valida: Requisito 1.5**
  - [x]* 10.3 Escrever testes de exemplo:
    - Sem nenhum campo de entrada → 422.
    - `include_pdf=False` → `pdfBase64` ausente ou null.
    - `include_pdf=True` → `pdfBase64` é string não vazia.
    - `jobId` inexistente → 404.
    - _Requisitos: 1.3, 1.4, 1.9, 1.11_

- [x] 11. Implementar serviço de argumentos contrários
  - [x] 11.1 Criar `api/services/counter_arguments_service.py` com:
    - Mapeamento `severidade → strength` e `tipo → category`.
    - Função `build_from_attention_points(attention_points, max_arguments) -> list[CounterArgument]`.
    - Função `build_with_llm(text, max_arguments, api_key, base_url, model) -> list[CounterArgument]` com prompt adversarial e parsing de JSON da resposta.
    - Função `generate_counter_arguments(text, attention_points, max_arguments, api_key, base_url, model) -> list[CounterArgument]`.
    - _Requisitos: 2.6, 2.7, 2.8, 2.11, 2.12_
  - [x]* 11.2 Escrever testes de propriedade em `tests/test_counter_arguments.py`:
    - **Propriedade 7: Fallback de contra-argumentos preserva campos obrigatórios**
    - Usar `hypothesis` para gerar listas de `attentionPoints` com campos variados.
    - **Valida: Requisitos 2.7, 2.8**
  - [x]* 11.3 Escrever testes de propriedade para `maxArguments`:
    - **Propriedade 6: `maxArguments` é respeitado como limite superior**
    - Para qualquer `maxArguments` em [1, 20] e qualquer lista de `attentionPoints`, verificar que `len(result) <= maxArguments`.
    - **Valida: Requisito 2.9**

- [x] 12. Implementar endpoint `POST /analyze/counter-arguments`
  - Adicionar rota `POST /analyze/counter-arguments` em `api/main.py`.
  - Implementar lógica de resolução de texto: `text` > `jobId` > `contractId`.
  - Chamar `generate_counter_arguments` do serviço.
  - Tratar erros: 422 para entrada ausente, 404 para `jobId` não encontrado.
  - _Requisitos: 2.1, 2.2, 2.3, 2.4, 2.5, 2.9, 2.10_
  - [x]* 12.1 Escrever testes de propriedade em `tests/test_counter_arguments.py`:
    - **Propriedade 5: Qualquer entrada válida produz resposta com schema correto**
    - Usar `hypothesis` para gerar combinações de `(text, maxArguments)`.
    - **Valida: Requisitos 2.2, 2.8, 2.10**
  - [x]* 12.2 Escrever testes de exemplo:
    - Sem nenhum campo de entrada → 422.
    - `maxArguments=0` → 422.
    - `maxArguments=21` → 422.
    - `attentionPoints` vazio + sem LLM → lista vazia, sem erro.
    - LLM falha → fallback aplicado silenciosamente.
    - _Requisitos: 2.3, 2.5, 2.11, 2.12_

- [x] 13. Checkpoint final — Garantir que todos os testes passam
  - Executar `pytest tests/ api/ml/tests/ -x --tb=short` e corrigir qualquer falha.
  - Verificar que os quatro novos endpoints respondem corretamente com `TestClient`.
  - Verificar que o pipeline inclui `urgency` no resultado e `durationMs` em todos os steps.
  - Garantir que todos os testes passam, perguntar ao usuário se houver dúvidas.

---

## Notas

- Tarefas marcadas com `*` são opcionais e podem ser puladas para um MVP mais rápido.
- Cada tarefa referencia os requisitos específicos que implementa para rastreabilidade.
- Os checkpoints (tarefas 5 e 13) garantem validação incremental.
- Os testes de propriedade usam `hypothesis` com `@settings(max_examples=100)`.
- Todos os novos endpoints seguem o padrão do projeto: sem autenticação, schemas Pydantic v2, `TestClient` com pipeline mockado.
