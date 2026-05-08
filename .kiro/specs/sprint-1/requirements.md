# Documento de Requisitos — Sprint 1 do JurisPro IA

## Introdução

Esta sprint entrega quatro funcionalidades incrementais ao microserviço FastAPI de análise jurídica com IA (JurisPro IA). O serviço já possui pipeline completo de NER, classificação, risco, sumarização, predição de desfecho, estimativa de honorários, busca de casos similares e comparação de contratos. A Sprint 1 adiciona:

1. **Resumo para cliente leigo** — endpoint que reescreve o `executive_summary` em linguagem acessível, com geração opcional de PDF.
2. **Simulação de argumentos contrários** — endpoint que gera lista de contra-argumentos adversariais a partir do texto ou de uma análise existente.
3. **Classificação de urgência processual** — módulo de regras heurísticas integrado ao pipeline e endpoint dedicado de triagem rápida.
4. **Latência por step no trace** — garantia de `durationMs` em todos os steps, adição de `startedAt`/`finishedAt` por step e endpoint de métricas de latência.

---

## Glossário

- **Sistema**: o microserviço FastAPI JurisPro IA.
- **Pipeline**: o orquestrador `AnalysisPipeline` em `api/ml/pipeline.py`.
- **TraceStep**: schema Pydantic que representa um passo do pipeline no trace de execução (`api/schemas/analysis.py`).
- **AnalysisResult**: modelo Pydantic de saída do pipeline (`api/ml/pipeline.py`).
- **UrgencyBlock**: novo bloco de dados com `score`, `level` e `rationale` para urgência processual.
- **LLM externo**: API compatível com OpenAI Chat Completions, configurada via `JURISPRO_OPENAI_API_KEY`.
- **Fallback de regras**: lógica determinística que opera sem LLM externo.
- **Buffer circular**: estrutura em memória de tamanho fixo (100 entradas) para armazenar métricas de latência por step.
- **jobId**: identificador único de uma execução de análise.
- **contractId**: identificador de contrato fornecido pelo cliente.
- **fpdf2**: biblioteca Python para geração de PDF, já presente em `requirements.txt`.
- **Nível de linguagem**: parâmetro `level` com valores `"leigo"`, `"intermediario"` ou `"tecnico"`.

---

## Requisitos

### Requisito 1.1 — Resumo para Cliente Leigo

**User Story:** Como advogado ou cliente, quero receber o resumo da análise jurídica em linguagem acessível ao público leigo, para que eu possa compartilhar o resultado com clientes sem formação jurídica.

#### Critérios de Aceitação

1. THE Sistema SHALL expor o endpoint `POST /analyze/summary/plain` sem autenticação.
2. WHEN o corpo da requisição contém `jobId`, `contractId` ou `text`, THE Sistema SHALL aceitar a requisição e processar o resumo.
3. IF nenhum dos campos `jobId`, `contractId` ou `text` for fornecido, THEN THE Sistema SHALL retornar HTTP 422 com mensagem de erro descritiva.
4. WHEN o parâmetro `level` não for fornecido, THE Sistema SHALL usar o valor padrão `"leigo"`.
5. IF o parâmetro `level` contiver valor diferente de `"leigo"`, `"intermediario"` ou `"tecnico"`, THEN THE Sistema SHALL retornar HTTP 422.
6. WHEN `JURISPRO_OPENAI_API_KEY` estiver configurado e `level` for `"leigo"` ou `"intermediario"`, THE Sistema SHALL invocar o LLM externo para reescrever o `executive_summary` em linguagem acessível ao nível solicitado.
7. WHEN `JURISPRO_OPENAI_API_KEY` não estiver configurado, THE Sistema SHALL aplicar simplificação por regras: remover jargão jurídico, encurtar frases longas e substituir termos técnicos por equivalentes comuns.
8. WHEN `include_pdf` for `true`, THE Sistema SHALL gerar um PDF com o texto do resumo usando `fpdf2` e retorná-lo codificado em base64 no campo `pdfBase64`.
9. WHEN `include_pdf` for `false` ou não fornecido, THE Sistema SHALL omitir o campo `pdfBase64` da resposta.
10. THE Sistema SHALL retornar resposta com os campos: `jobId` (string ou null), `summaryText` (string), `level` (string), `generatedAt` (ISO 8601 UTC) e `pdfBase64` (string ou null).
11. WHEN a geração do PDF falhar, THE Sistema SHALL retornar HTTP 500 com código de erro `PDF_GENERATION_FAILED` e `pdfBase64: null`.
12. WHEN o LLM externo falhar, THE Sistema SHALL aplicar o fallback de regras e retornar o resumo sem indicar erro ao cliente.
13. THE Sistema SHALL processar a requisição em menos de 30 segundos quando o LLM externo estiver disponível.
14. THE Sistema SHALL processar a requisição em menos de 2 segundos quando operar apenas com fallback de regras.

---

### Requisito 1.2 — Simulação de Argumentos Contrários

**User Story:** Como advogado, quero receber uma lista de argumentos que a parte contrária poderia usar, para que eu possa preparar a defesa ou a estratégia processual com antecedência.

#### Critérios de Aceitação

1. THE Sistema SHALL expor o endpoint `POST /analyze/counter-arguments` sem autenticação.
2. WHEN o corpo da requisição contém ao menos um de `text`, `jobId` ou `contractId`, THE Sistema SHALL aceitar a requisição.
3. IF nenhum dos campos `text`, `jobId` ou `contractId` for fornecido, THEN THE Sistema SHALL retornar HTTP 422 com mensagem de erro descritiva.
4. WHEN `maxArguments` não for fornecido, THE Sistema SHALL usar o valor padrão `5`.
5. IF `maxArguments` for menor que 1 ou maior que 20, THEN THE Sistema SHALL retornar HTTP 422.
6. WHEN `JURISPRO_OPENAI_API_KEY` estiver configurado, THE Sistema SHALL enviar prompt adversarial ao LLM externo e retornar a lista de contra-argumentos gerada.
7. WHEN `JURISPRO_OPENAI_API_KEY` não estiver configurado, THE Sistema SHALL reformular os `attentionPoints` existentes como argumentos adversariais usando regras determinísticas.
8. THE Sistema SHALL retornar cada argumento com os campos: `text` (string), `strength` (um de `"forte"`, `"medio"`, `"fraco"`) e `category` (string).
9. THE Sistema SHALL retornar no máximo `maxArguments` argumentos na lista.
10. THE Sistema SHALL retornar resposta com os campos: `jobId` (string ou null), `arguments` (lista), `generatedAt` (ISO 8601 UTC).
11. WHEN o LLM externo falhar, THE Sistema SHALL aplicar o fallback de regras e retornar a lista sem indicar erro ao cliente.
12. WHEN a lista de `attentionPoints` estiver vazia e o LLM não estiver configurado, THE Sistema SHALL retornar lista vazia de argumentos sem erro.

---

### Requisito 1.3 — Classificação de Urgência Processual

**User Story:** Como advogado, quero que o sistema classifique automaticamente a urgência de um documento processual, para que eu possa priorizar minha fila de trabalho sem ler cada peça integralmente.

#### Critérios de Aceitação

1. THE Sistema SHALL implementar o módulo `api/ml/models/urgency_classifier.py` com função `classify(text, entities_datas, contract_type)` que retorna `score` (int 0–100), `level` (string) e `rationale` (string).
2. WHEN o texto contiver palavras-chave de urgência imediata (`"liminar"`, `"tutela de urgência"`, `"tutela antecipada"`, `"prazo fatal"`, `"audiência"`, `"penhora"`, `"arresto"`, `"sequestro"`, `"busca e apreensão"`), THE Classificador SHALL atribuir `level = "IMEDIATO"` e `score >= 80`.
3. WHEN o texto contiver palavras-chave de urgência alta (`"prazo"`, `"intimação"`, `"citação"`, `"recurso"`, `"apelação"`, `"embargos"`), THE Classificador SHALL atribuir `level = "URGENTE"` e `score` entre 50 e 79.
4. WHEN `entities_datas` contiver ao menos uma data dentro de 7 dias corridos a partir da data atual, THE Classificador SHALL atribuir `level = "IMEDIATO"` e `score >= 80`, independentemente das palavras-chave.
5. WHEN `entities_datas` contiver ao menos uma data entre 8 e 30 dias corridos a partir da data atual e nenhuma data dentro de 7 dias, THE Classificador SHALL atribuir `level = "URGENTE"` e `score` entre 50 e 79.
6. WHEN não houver palavras-chave de urgência e não houver datas próximas, THE Classificador SHALL atribuir `level = "NORMAL"` e `score` entre 20 e 49.
7. WHEN o texto for muito curto (menos de 50 palavras) ou não contiver indicadores processuais, THE Classificador SHALL atribuir `level = "BAIXO"` e `score < 20`.
8. THE Classificador SHALL preencher o campo `rationale` com texto explicativo indicando quais critérios determinaram o nível de urgência.
9. THE Pipeline SHALL incluir o campo `urgency: UrgencyBlock` no `AnalysisResult` após a execução do step `risk_scoring`.
10. THE Sistema SHALL expor o endpoint `POST /analyze/urgency` sem autenticação, que aceita `text` no corpo e retorna `{ score, level, rationale, generatedAt }` sem executar o pipeline completo.
11. WHEN o campo `text` do endpoint `/analyze/urgency` estiver vazio ou ausente, THE Sistema SHALL retornar HTTP 422.
12. THE Sistema SHALL registrar o step `urgency_classification` no trace do pipeline com `durationMs` preenchido.

---

### Requisito 1.4 — Latência por Step no Trace

**User Story:** Como operador do sistema, quero visualizar a latência individual de cada step do pipeline e métricas agregadas, para que eu possa identificar gargalos e otimizar o desempenho.

#### Critérios de Aceitação

1. THE Pipeline SHALL garantir que todos os steps registrados em `_record_step` tenham `durationMs` preenchido com valor inteiro não negativo (nunca `None` ou `null`).
2. THE Sistema SHALL adicionar os campos `startedAt` (ISO 8601 UTC) e `finishedAt` (ISO 8601 UTC) ao schema `TraceStep` em `api/schemas/analysis.py`.
3. THE Pipeline SHALL preencher `startedAt` e `finishedAt` em cada chamada a `_record_step`.
4. THE Sistema SHALL manter um buffer circular em memória com capacidade para as últimas 100 execuções do pipeline, armazenando a duração de cada step por nome.
5. THE Sistema SHALL expor o endpoint `GET /metrics/pipeline` sem autenticação.
6. WHEN o endpoint `GET /metrics/pipeline` for chamado, THE Sistema SHALL retornar a lista de steps com `avgDurationMs`, `p95DurationMs` e `callCount` para cada step registrado no buffer.
7. THE Sistema SHALL retornar o campo `collectedAt` (ISO 8601 UTC) na resposta de `/metrics/pipeline`.
8. WHEN o buffer estiver vazio, THE Sistema SHALL retornar lista `steps` vazia e `collectedAt` com o timestamp atual.
9. THE Sistema SHALL calcular `p95DurationMs` como o percentil 95 das durações registradas para cada step.
10. WHEN um novo resultado de pipeline for produzido, THE Sistema SHALL atualizar o buffer circular de métricas de forma thread-safe.
11. THE Sistema SHALL retornar o schema `{ steps: [{ step, avgDurationMs, p95DurationMs, callCount }], collectedAt }` no endpoint `/metrics/pipeline`.
