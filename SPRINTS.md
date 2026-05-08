# JurisPro IA — Sprints de Desenvolvimento

> Baseado na análise de `ANALISE_ARQUITETURAL.md` cruzada com o código atual.

---

## ✅ O que já está entregue

| Funcionalidade | Onde está no código |
|---|---|
| Extração de texto (PDF/DOCX/TXT + OCR) | `api/ml/text_extractor.py` |
| Classificação de tipo/área jurídica | `api/ml/models/classifier.py` |
| Análise de risco (BERT + regras heurísticas) | `api/ml/models/risk_analyzer.py` |
| NER (pessoas, orgs, legislação, datas, valores) | `api/ml/models/ner.py` |
| Sumarização (T5 + extractiva) | `api/ml/models/summarizer.py` |
| Predição de desfecho (win predictor) | `api/ml/models/win_predictor.py` |
| Estimativa de honorários | `api/ml/models/fee_estimator.py` |
| Busca de casos similares (Qdrant) | `api/ml/models/case_retriever.py` |
| Comparação de versões de contrato | `api/ml/models/contract_differ.py` |
| Detecção de tipo de documento | `api/ml/document_kind.py` |
| Enriquecimento via LLM externo (OpenAI-compat) | `api/ml/external_llm.py` |
| Pipeline orquestrador com modos fast/standard/deep | `api/ml/pipeline.py` |
| Endpoints REST (file, text, async, jobs, compare) | `api/main.py` |
| Cache em memória com TTL e LRU | `api/main.py` |
| Fila assíncrona em memória (thread pool) | `api/main.py` |
| Trace de pipeline com steps e custo estimado | `api/main.py` |
| Extração de cláusulas, timeline e seções | `api/ml/preprocessor.py` |
| Regras heurísticas (LGPD, multa, rescisão, PI, foro) | `api/ml/models/risk_analyzer.py` |
| Auth, routers, schemas, models DB | `api/routers/`, `api/schemas/`, `api/models/` |
| Docker + docker-compose | `Dockerfile`, `docker-compose.yml` |

---

## 🔴 Sprint 1 — Infraestrutura e Quick Wins (Semanas 1–3)

> Foco: estabilizar a base e entregar valor imediato com baixo esforço.

### 1.1 Migração da fila em memória → Redis/BullMQ

**Por que agora:** bloqueador explícito de escala multi-tenant. Sem isso, qualquer deploy com mais de 1 worker perde jobs.

- [x] Adicionar `redis` e `celery` (ou `arq`) ao `requirements.txt`
- [x] Criar `api/worker.py` com worker Celery/arq
- [x] Substituir `_run_analysis_job_in_thread` por task assíncrona no broker
- [x] Atualizar `docker-compose.yml` com serviço Redis
- [x] Manter fallback em memória via env `JURISPRO_QUEUE_BACKEND=memory` para dev local
- [x] Testes de integração: enfileirar → processar → polling retorna `done`

### 1.2 Resumo executivo para cliente leigo (feature 3.4)

**Complexidade:** Baixa. **Impacto:** Alto. Estimativa: 1–2 dias.

- [x] Novo endpoint `POST /analyze/summary/plain` que recebe `jobId` ou `contractId`
- [x] Parâmetro `level: "leigo" | "intermediario" | "tecnico"` (default `leigo`)
- [x] Usar LLM externo (se configurado) ou template de regras para reescrever `executive_summary`
- [x] Retornar PDF gerado com branding configurável (lib `reportlab` ou `weasyprint`)
- [x] Schema de resposta: `{ summaryText, level, generatedAt, pdfBase64? }`

### 1.3 Simulação de argumentos contrários (feature 2.3)

**Complexidade:** Baixa. **Impacto:** Alto. Estimativa: 2–3 dias.

- [x] Novo endpoint `POST /analyze/counter-arguments`
- [x] Body: `{ text?, jobId?, contractId?, maxArguments: 5 }`
- [x] Prompt estruturado: recebe peça/contrato → LLM gera lista de contra-argumentos com força estimada
- [x] Fallback sem LLM: retornar lista baseada nos `attentionPoints` existentes reformulados como argumentos adversariais
- [x] Schema: `{ arguments: [{ text, strength: "forte"|"médio"|"fraco", category }] }`

### 1.4 Latência por step no `trace`

**Complexidade:** Mínima (já existe estrutura). **Impacto:** Operacional.

- [x] Confirmar que todos os steps em `pipeline.py` registram `durationMs` individualmente (hoje alguns têm `None`)
- [x] Adicionar `startedAt` e `finishedAt` por step no trace
- [x] Expor endpoint `GET /metrics/pipeline` com médias de latência por step (últimas N execuções em memória)

---

## 🟡 Sprint 2 — Diferenciais Competitivos Parte 1 (Semanas 4–7)

### 2.1 Classificação de urgência processual (feature 1.4)

**Complexidade:** Baixa-média. **Impacto:** Alto.

- [x] Novo modelo leve de classificação multi-label em `api/ml/models/urgency_classifier.py`
- [x] Features: tipo de ato processual (do classifier existente) + datas extraídas (NER) + palavras-chave de prazo
- [x] Score de urgência 0–100 + label `IMEDIATO | URGENTE | NORMAL | BAIXO`
- [x] Integrar ao pipeline principal: novo campo `urgency` no `AnalysisResult`
- [x] Expor no contrato JSON: `result.urgency.score`, `result.urgency.level`, `result.urgency.rationale`
- [x] Endpoint dedicado `POST /analyze/urgency` para triagem rápida sem análise completa

### 2.2 Checklist de compliance automático (feature 3.2)

**Complexidade:** Média. **Impacto:** Alto.

- [x] Criar `api/ml/models/compliance_checker.py`
- [x] Checklists estáticos por regulação: LGPD, CDC, CLT, CPC (YAML/JSON em `api/data/checklists/`)
- [x] Modelo local classifica área → seleciona checklist → verifica cada item contra o texto (regex + NLI leve)
- [x] LLM externo (se disponível) verifica itens ambíguos
- [x] Schema de resposta: `{ regulation, items: [{ id, description, status: "ok"|"fail"|"warning"|"na", evidence }] }`
- [x] Integrar ao pipeline: campo `compliance` no resultado quando `mode=deep`

### 2.3 Detecção de cláusulas abusivas — fine-tune local (feature 1.1)

**Complexidade:** Média. **Impacto:** Alto. **Diferencial:** Alto.

- [x] Script de preparação de dataset em `scripts/prepare_clause_dataset.py`
- [ ] Fine-tune de DeBERTa-v3-small em corpus de cláusulas rotuladas (abusiva/padrão/favorável)
- [ ] Salvar pesos em `hf_models/classificacao_clausulas/`
- [x] Novo módulo `api/ml/models/clause_classifier.py`
- [x] Pipeline: segmentar contrato em cláusulas (já existe em `preprocessor.extract_clauses`) → classificar cada uma
- [x] Integrar ao `attentionPoints` com `source: "clause_classifier"` e evidência textual
- [x] Testes: `api/ml/tests/test_clause_classifier.py`

### 2.4 Extração de obrigações e prazos → agenda automática (feature 1.2)

**Complexidade:** Média-alta.

- [x] Novo módulo `api/ml/models/obligation_extractor.py`
- [x] NER especializado para triplas `(sujeito, obrigação, prazo)` usando SpanBERT ou pipeline de RE
- [x] Tratar prazos implícitos: "30 dias após a assinatura" → resolver data absoluta quando `entities.datas` tiver data de referência
- [x] Schema de saída: `{ obligations: [{ subject, obligation, deadline, deadlineAbsolute?, confidence }] }`
- [x] Novo endpoint `POST /analyze/obligations`
- [x] Integrar ao pipeline principal: campo `obligations` no resultado
- [x] Webhook opcional para NestJS criar eventos de calendário

---

## 🟡 Sprint 3 — Diferenciais Competitivos Parte 2 (Semanas 8–12)

### 3.1 Score de qualidade da peça antes de protocolar (feature 5.3)

**Complexidade:** Média-alta. **Impacto:** Muito alto. **Diferencial:** Muito alto.

- [x] Novo módulo `api/ml/models/quality_scorer.py`
- [x] Critérios multi-dimensionais:
  - Completude estrutural (seções obrigatórias presentes — regras)
  - Coerência argumentativa (NLI entre premissas e conclusão)
  - Citações verificáveis (Qdrant: legislação mencionada existe no índice?)
  - Linguagem técnica adequada (classificador de registro)
- [x] Score 0–100 com breakdown por dimensão + sugestões específicas
- [x] Endpoint `POST /analyze/quality`
- [x] Schema: `{ score, dimensions: { completeness, coherence, citations, language }, suggestions: [{ dimension, issue, fix }] }`

### 3.2 Monitor de jurisprudência com alertas (feature 5.2)

**Complexidade:** Média-alta. **Impacto:** Muito alto.

- [x] Script de ingestão contínua `scripts/ingest_jurisprudencia.py` (STJ, STF, TJs via API pública)
- [x] Job periódico (cron/Celery beat) que ingere novas decisões → gera embeddings → insere no Qdrant
- [x] Endpoint `POST /monitor/subscribe` — registra `{ caseId, contractId, threshold: 0.85 }`
- [x] Job de matching: compara novas decisões com casos ativos → dispara webhook NestJS quando `similarity > threshold`
- [x] Schema de notificação: `{ caseId, newDecisionId, similarity, tribunal, summary, url }`
- [x] Endpoint `GET /monitor/alerts/{caseId}` para polling

### 3.3 Geração de minutas com LLM (feature 2.1)

**Complexidade:** Média. **Impacto:** Alto. **Maior willingness-to-pay do mercado.**

- [x] Novo endpoint `POST /generate/draft`
- [x] Body: `{ documentType: "peticao_inicial"|"contestacao"|"contrato"|..., context: { parties, subject, facts, claims }, style?: "formal"|"conciso", firmId? }`
- [x] Prompt engineering estruturado por tipo de peça (templates em `api/data/templates/`)
- [x] Few-shot com peças anteriores do escritório quando `firmId` fornecido (RAG privado — depende da Sprint 4)
- [x] Retornar minuta com seções marcadas `[REVISAR]` onde confiança é baixa
- [x] Schema: `{ draft, sections: [{ title, content, needsReview, confidence }], disclaimer }`
- [x] Disclaimer obrigatório: revisão humana necessária

### 3.4 Cache semântico antes do LLM (feature arquitetural)

**Complexidade:** Baixa. **Impacto:** Redução de custo 30–60%.**

- [x] Antes de chamar LLM externo, verificar no Qdrant se existe análise recente semanticamente idêntica
- [x] Chave: embedding do `rep` (chunk representativo) + `mode` + `contract_type`
- [x] Threshold de similaridade configurável via env `JURISPRO_SEMANTIC_CACHE_THRESHOLD=0.96`
- [x] Coleção dedicada no Qdrant: `semantic_cache` com TTL de 7 dias
- [x] Métricas: `cache_hit_rate` exposto em `/metrics/pipeline`

### 3.5 Pipeline paralelo de micro-tarefas (feature arquitetural)

**Complexidade:** Média. **Impacto:** Latência percebida.

- [x] Refatorar `pipeline.analyze()` para executar steps independentes em paralelo com `asyncio.gather` ou `ThreadPoolExecutor`
- [x] Steps paralelizáveis: NER, classificação, risco, Qdrant search (todos leem o mesmo texto, não dependem entre si)
- [x] Steps sequenciais: chunk_document → [paralelo] → sumarização (usa resultado da classificação) → validate_output
- [x] Streaming de resultados parciais via SSE: endpoint `POST /analyze/file/stream` retorna eventos à medida que cada step completa
- [x] Testes de regressão: resultado final deve ser idêntico ao pipeline sequencial

---

## 🔵 Sprint 4 — Enterprise e Lock-in (Semanas 13–20)

### 4.1 RAG privado por escritório — Qdrant multi-tenant (feature 4.1)

**Complexidade:** Alta. **Diferencial:** Muito alto (lock-in).

- [x] Estratégia de isolamento: uma collection Qdrant por `firmId` (`firm_{uuid}`)
- [x] Pipeline de ingestão: `POST /firms/{firmId}/knowledge/ingest` — recebe peças aprovadas, pareceres, modelos internos
- [x] Geração de embeddings e indexação assíncrona (Celery)
- [x] Integrar ao `case_retriever.py`: quando `firm_id` fornecido, busca primeiro na collection privada, depois na pública
- [x] Integrar à geração de minutas (Sprint 3.3): few-shot com peças do escritório
- [x] Endpoint `GET /firms/{firmId}/knowledge/stats` — documentos indexados, última atualização
- [x] Segurança: validar `firmId` contra token JWT antes de qualquer operação

### 4.2 Auditoria e rastreabilidade de decisões IA (feature 4.2)

**Complexidade:** Média.

- [ ] Expandir `trace` com: versão exata do modelo (hash dos pesos), features que mais influenciaram o score (SHAP/attention weights), fontes jurisprudenciais usadas (IDs do Qdrant), hash SHA-256 do documento analisado
- [ ] Endpoint `GET /audit/{jobId}` retorna trace completo imutável
- [ ] Armazenamento: NestJS persiste o trace no PostgreSQL (este serviço apenas gera)
- [ ] Formato de exportação: JSON-LD para compatibilidade com sistemas de compliance

### 4.3 Análise de portfólio de contratos (feature 4.4)

**Complexidade:** Média.

- [ ] Endpoint `POST /portfolio/analyze` — recebe lista de `contractId`s já analisados
- [ ] Job batch que agrega scores: distribuição de risco, alertas críticos por área, contratos próximos do vencimento
- [ ] Dashboard JSON: `{ riskDistribution, criticalAlerts, expiringContracts, topRisks, byArea }`
- [ ] Endpoint `GET /portfolio/dashboard/{firmId}` com cache de 1h

### 4.4 Assistente de negociação contratual (feature 3.1)

**Complexidade:** Média. **Diferencial:** Muito alto.

- [ ] Depende de: cláusulas abusivas (Sprint 2.3) + LLM (Sprint 3.3)
- [ ] Endpoint `POST /negotiate/suggest`
- [ ] Fluxo: modelo local detecta cláusulas problemáticas → LLM gera 2–3 redações alternativas por cláusula com justificativa jurídica
- [ ] Schema: `{ suggestions: [{ originalClause, alternatives: [{ text, rationale, riskReduction }] }] }`

### 4.5 Parecer fundamentado com RAG (feature 2.2)

**Complexidade:** Média-alta. **Diferencial:** Muito alto.

- [ ] Depende de: RAG privado (4.1) + Qdrant público já existente
- [ ] Endpoint `POST /generate/opinion`
- [ ] Fluxo: Qdrant recupera jurisprudência relevante → LLM sintetiza parecer citando decisões reais (IDs rastreáveis)
- [ ] Citações com link para fonte original (não inventadas)
- [ ] Schema: `{ opinion, citations: [{ id, tribunal, summary, relevance }], disclaimer }`

---

## 🟣 Sprint 5 — Inovação e Moat (Semanas 21–32)

### 5.1 Grafo de conhecimento jurídico do caso (feature 6.1)

**Complexidade:** Alta.

- [ ] Módulo de extração de relações (RE) em `api/ml/models/relation_extractor.py`
- [ ] Construção de grafo: entidades NER como nós, relações como arestas (NetworkX ou Neo4j)
- [ ] Endpoint `GET /cases/{caseId}/graph` retorna grafo em formato JSON (nodes + edges)
- [ ] Formato compatível com D3.js / Cytoscape.js para visualização no frontend

### 5.2 Detecção de inconsistências internas (feature 1.3)

**Complexidade:** Média.

- [ ] Módulo `api/ml/models/consistency_checker.py`
- [ ] Combinar entidades NER com modelo NLI para detectar contradições entre pares de sentenças
- [ ] Controle de falsos positivos: threshold calibrado + score de confiança por contradição
- [ ] Integrar ao pipeline: campo `inconsistencies` no resultado quando `mode=deep`

### 5.3 A/B testing de modelos por tenant (feature arquitetural)

**Complexidade:** Média.

- [ ] Feature flags no NestJS passam `modelVariant` para o FastAPI via header ou form field
- [ ] Pipeline lê `modelVariant` e carrega pesos alternativos de `hf_models/{model}_{variant}/`
- [ ] Registro no `trace`: qual variante foi usada
- [ ] Endpoint `POST /admin/model-variants` para registrar variantes disponíveis

### 5.4 Análise de viabilidade econômica (feature 3.3)

**Complexidade:** Média.

- [ ] Depende de: win predictor (já existe) + fee estimator (já existe) + LLM
- [ ] Endpoint `POST /analyze/viability`
- [ ] Fluxo: modelo local estima probabilidade de desfecho + extrai valor da causa → LLM calcula cenários (honorários × probabilidade × custas estimadas)
- [ ] Schema: `{ viability: "recomendado"|"neutro"|"desaconselhado", scenarios: [{ name, probability, netValue, rationale }] }`

### 5.5 Simulação de audiência adversarial (feature 6.3)

**Complexidade:** Alta.

- [ ] Endpoint `POST /simulate/hearing` — inicia sessão conversacional
- [ ] LLM em modo juiz/parte contrária faz perguntas baseadas no processo
- [ ] Endpoint `POST /simulate/hearing/{sessionId}/respond` — advogado responde, LLM avalia e sugere melhorias
- [ ] Schema de sessão: `{ sessionId, role: "juiz"|"parte_contraria", question, evaluation?, suggestions? }`

---

## 📊 Resumo por Sprint

| Sprint | Semanas | Foco | Features |
|---|---|---|---|
| **Sprint 1** | 1–3 | Infraestrutura + Quick Wins | Redis/fila, resumo leigo, contra-argumentos, trace latência |
| **Sprint 2** | 4–7 | Diferenciais Parte 1 | Urgência, compliance, cláusulas abusivas, obrigações/prazos |
| **Sprint 3** | 8–12 | Diferenciais Parte 2 | Score qualidade, monitor jurisprudência, minutas, cache semântico, pipeline paralelo |
| **Sprint 4** | 13–20 | Enterprise | RAG privado, auditoria, portfólio, negociação, parecer RAG |
| **Sprint 5** | 21–32 | Inovação | Grafo, inconsistências, A/B testing, viabilidade, simulação audiência |

---

## 🏗️ Dependências entre features

```
Redis (1.1) ──────────────────────────────────────────────────────► Monitor jurisprudência (3.2)
                                                                      Portfólio batch (4.3)

Cláusulas abusivas (2.3) ────────────────────────────────────────► Negociação contratual (4.4)

LLM externo (já existe) ─────────────────────────────────────────► Minutas (3.3)
                                                                      Negociação (4.4)
                                                                      Parecer RAG (4.5)
                                                                      Viabilidade (5.4)

RAG privado (4.1) ───────────────────────────────────────────────► Minutas com estilo escritório (3.3)
                                                                      Parecer RAG (4.5)

NER (já existe) + RE (5.1) ──────────────────────────────────────► Grafo de conhecimento (5.1)

Win predictor (já existe) + Fee estimator (já existe) ───────────► Viabilidade econômica (5.4)
```

---

## ⚠️ Débitos técnicos a resolver em paralelo

| Débito | Impacto | Sprint sugerida |
|---|---|---|
| ~~Fila em memória não escala para múltiplos workers~~ | Resolvido com backend Celery/Redis + fallback memory | Sprint 1 |
| ~~`durationMs: null` em alguns steps do trace~~ | Resolvido com `durationMs` obrigatório e timestamps por step | Sprint 1 |
| Chunking de documentos longos pode truncar (sem chunking hierárquico) | Qualidade em contratos complexos | Sprint 2 |
| Modelos carregados em singleton compartilhado — sem isolamento de tenant | Risco de vazamento de contexto | Sprint 4 |
| Sem rate limiting por tenant na API FastAPI | Risco de abuso | Sprint 4 |
| `CORS_ORIGINS=*` no padrão | Segurança em produção | Imediato (config) |
