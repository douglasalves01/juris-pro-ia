# JurisPro IA — O que já entregamos para o escritório

> Visão consolidada das funcionalidades de análise disponíveis hoje para advogados e escritórios.

---

## Análise de documentos jurídicos

O núcleo da plataforma. O advogado envia um arquivo (PDF, DOCX ou TXT) ou texto puro e recebe um laudo estruturado com tudo abaixo.

### O que o laudo inclui

| Funcionalidade | O que o advogado recebe |
|---|---|
| **Extração de texto** | Leitura automática de PDF (inclusive PDF escaneado via OCR), DOCX e TXT |
| **Classificação do documento** | Tipo (contrato, petição, contestação, etc.) e área jurídica (trabalhista, cível, LGPD, etc.) |
| **Análise de risco** | Score de risco 0–100, nível (BAIXO / MÉDIO / ALTO / CRÍTICO) e pontos de atenção com evidência textual |
| **NER — Entidades nomeadas** | Pessoas, organizações, legislação citada, datas e valores monetários extraídos automaticamente |
| **Sumarização** | Resumo executivo do documento em linguagem técnica |
| **Predição de desfecho** | Estimativa de probabilidade de êxito com base no tipo de ação e área |
| **Estimativa de honorários** | Faixa de honorários estimada com base no valor da causa e complexidade |
| **Busca de casos similares** | Jurisprudência e casos análogos recuperados do índice vetorial (Qdrant) |
| **Comparação de versões** | Diff estruturado entre duas versões de um contrato (cláusulas adicionadas, removidas, alteradas) |
| **Extração de cláusulas** | Lista de cláusulas identificadas no contrato com posição no texto |
| **Timeline do documento** | Linha do tempo de eventos e datas extraídos do texto |
| **Detecção de tipo de documento** | Identificação automática do formato antes de processar |

---

## Análises especializadas (endpoints dedicados)

Além do laudo completo, o advogado pode acionar análises pontuais sem precisar reprocessar o documento inteiro.

### Urgência processual

Classifica o documento em uma fila de prioridade:

- Score de urgência de 0 a 100
- Nível: **IMEDIATO**, **URGENTE**, **NORMAL** ou **BAIXO**
- Justificativa textual do score

Útil para triagem de fila quando chegam vários documentos ao mesmo tempo.

---

### Checklist de compliance automático

Verifica conformidade com as principais regulações brasileiras:

- **LGPD** — Lei Geral de Proteção de Dados
- **CDC** — Código de Defesa do Consumidor
- **CLT** — Consolidação das Leis do Trabalho
- **CPC** — Código de Processo Civil

Para cada item do checklist, o sistema retorna:
- Status: `ok`, `falha`, `atenção` ou `não aplicável`
- Evidência textual que embasou a classificação

---

### Detecção de cláusulas abusivas

Analisa cada cláusula do contrato individualmente e classifica como:

- **Abusiva** — potencialmente prejudicial ao cliente
- **Padrão** — dentro do esperado para o tipo de contrato
- **Favorável** — benéfica para a parte representada

Cada classificação vem com score de confiança e trecho do texto como evidência.

---

### Extração de obrigações e prazos

Extrai triplas estruturadas `(sujeito, obrigação, prazo)` do documento:

- Prazos explícitos e implícitos ("30 dias após a assinatura" → data absoluta calculada)
- Saída pronta para integração com agenda/calendário
- Webhook opcional para criar eventos automaticamente no sistema do escritório

---

### Score de qualidade da peça

Avalia a peça jurídica antes de protocolar em quatro dimensões:

| Dimensão | O que avalia |
|---|---|
| Completude estrutural | Seções obrigatórias presentes |
| Coerência argumentativa | Premissas e conclusão consistentes |
| Citações verificáveis | Legislação mencionada existe no índice |
| Linguagem técnica | Registro adequado ao tipo de peça |

Retorna score 0–100 com sugestões específicas de melhoria por dimensão.

---

### Resumo para cliente leigo

Reescreve o resumo executivo em linguagem acessível, com três níveis configuráveis:

- **Leigo** — sem jargão jurídico
- **Intermediário** — linguagem simplificada com termos explicados
- **Técnico** — padrão jurídico (equivalente ao laudo original)

Pode gerar PDF com branding do escritório para enviar diretamente ao cliente.

---

### Simulação de contra-argumentos

Recebe a peça ou contrato e gera os melhores argumentos que a parte contrária poderia usar, com:

- Texto do argumento
- Força estimada: **forte**, **médio** ou **fraco**
- Categoria jurídica do argumento

Útil para preparação estratégica antes de audiências ou negociações.

---

## Geração de minutas

Gera o primeiro rascunho de peças jurídicas a partir do contexto do caso:

- Tipos suportados: petição inicial, contestação, contrato e outros
- Seções marcadas com `[REVISAR]` onde a confiança é baixa
- Disclaimer obrigatório de revisão humana incluído na resposta
- Estilo configurável: formal ou conciso

---

## Monitor de jurisprudência

Mantém o advogado atualizado quando novas decisões impactam casos ativos:

- Subscrição por caso ou contrato com threshold de similaridade configurável
- Matching automático de novas decisões contra casos cadastrados
- Notificação via webhook quando similaridade supera o threshold
- Endpoint de polling para consultar alertas por caso

---

## Infraestrutura de análise

### Modos de análise

| Modo | Velocidade | Profundidade |
|---|---|---|
| `fast` | Segundos | Classificação + risco básico |
| `standard` | ~10–30s | Laudo completo |
| `deep` | ~30–60s | Laudo completo + compliance + cláusulas |

### Processamento assíncrono

- Envio do documento → recebe `jobId` imediatamente
- Polling de status: `queued → processing → done`
- Suporte a fila RabbitMQ com worker dedicado

### Cache inteligente

- Cache em memória com TTL de 1 hora para documentos idênticos
- Cache semântico no Qdrant para documentos similares (evita rechamadas ao LLM)
- Redução de custo estimada de 30–60% em escritórios com contratos padronizados

### Rastreabilidade

Cada análise retorna um objeto `trace` com:

- Versão do pipeline
- Timestamps de início e fim
- Duração por step do pipeline
- Custo estimado de API (quando LLM externo é usado)
- Indicação de qual provider foi utilizado em cada etapa

---

## Segurança e multi-tenant

- Autenticação via JWT com validação de `firm_id` em todas as operações
- Isolamento de dados por escritório (collections Qdrant separadas por `firmId`)
- Ingestão de base de conhecimento privada por escritório (peças aprovadas, modelos internos)
- Rate limiting e controle de acesso por papel: `admin`, `advogado`, `secretaria`

---

## Formatos de entrada suportados

| Formato | Observação |
|---|---|
| PDF com texto selecionável | Extração direta |
| PDF escaneado | OCR automático via Tesseract |
| DOCX | Extração nativa |
| TXT / texto puro | Via body JSON |

Limite de upload: 50 MB por arquivo.

---

## O que ainda não está disponível

As funcionalidades abaixo estão planejadas mas ainda não entregues:

- Grafo de conhecimento jurídico do caso (visualização de relações entre partes, contratos e processos)
- Detecção de inconsistências internas no documento (contradições entre cláusulas)
- Simulação de audiência adversarial (treino com IA no papel de juiz ou parte contrária)
- Análise de portfólio consolidado de contratos (dashboard de risco agregado)
- Parecer fundamentado com citações rastreáveis (RAG com jurisprudência real)
- Assistente de negociação contratual (sugestões de redação alternativa para cláusulas problemáticas)
- Análise de viabilidade econômica da ação (custo-benefício antes de aceitar o caso)
- A/B testing de modelos por tenant

---

*Documento gerado em maio de 2026 com base no estado atual do repositório.*
