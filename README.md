# JurisPro IA

Camada de **inteligência artificial para documentos jurídicos em português**. Este repositório é o **microserviço FastAPI** que o backend **NestJS** chama para extrair texto, classificar, resumir, estimar risco e honorários, recuperar jurisprudência semanticamente parecida e devolver um **JSON único** pronto para o produto (Next.js → NestJS → **este serviço**).

**Escopo consciente:** não há autenticação nem PostgreSQL aqui. Clientes, processos do escritório, vínculo advogado–caso e métricas de negócio ficam no NestJS; a IA roda stateless com cache opcional em memória.

---

## Onde este serviço encaixa no produto

```
Next.js (experiência do advogado)
        ↓
NestJS (auth, escritório, LGPD, PostgreSQL, filas de negócio)
        ↓
JurisPro IA — FastAPI (análise de texto / arquivo)
        ↓
Qdrant (somente jurisprudência / decisões públicas indexadas)
```

**Integrações que o projeto deve respeitar**

| Integração | Responsabilidade |
|------------|------------------|
| **NestJS** | Propaga `jobId`, `contractId`, `regiao`, `mode`; persiste resultado e exibe ao usuário; trata erros do contrato (`status`, `error.code`). |
| **Qdrant** | Busca por similaridade semântica sobre coleção de decisões públicas (ex.: `casos_juridicos`). Sem dados de clientes. |
| **Modelos locais (`hf_models/`)** | NER, classificação, risco, sumarização T5, predição de desfecho, honorários, embeddings. |
| **LLM opcional (OpenAI-compatible)** | Enriquecimento de parecer quando `JURISPRO_OPENAI_API_KEY` está definida e o modo/gatilhos permitem. |
| **Armazenamento de arquivo** | Idealmente no NestJS/object storage; este serviço pode receber upload direto ou texto já extraído. |

Contrato JSON documentado em `api/schemas/analysis.py` e `schemas/analysis-response.schema.json`.

---

## O que o sistema entrega hoje

Para **contratos, PDFs, petições e demais textos jurídicos** (PDF / DOCX / TXT ou corpo JSON com texto):

- **Resumo e parecer preliminar** — sumarização local (T5) conforme `mode`; texto complementar via LLM quando configurado.
- **Classificação de área e tipo de documento** — apoio à triagem (consumidor, trabalhista, etc.; tipo estrutural petição/sentença/contrato…).
- **Risco** — score 0–100 e nível normalizado no contrato (`BAIXO` … `CRITICO`).
- **Pontos de atenção** — alertas com severidade (incluindo críticos), descrição e evidência textual quando disponível.
- **Pontos positivos e recomendações** — gerados no pipeline interno e refletidos no bloco de parecer / análise enviado ao cliente.
- **Probabilidade de desfecho** — valor + confiança + racional; combina **regras** (ex.: trechos decisórios explícitos) e **modelo**; **não substitui julgamento humano nem prevê sentença**.
- **Casos semelhantes** — lista para o advogado usar como **referência de linha jurisprudencial** (ementas públicas indexadas), não “outros contratos iguais ao seu”.
- **Estimativa de honorários** — `min` / `suggested` / `max` e racional, usando **região** enviada pelo NestJS (`regiao`).
- **Entidades** — pessoas, organizações, legislação, datas, valores monetários (NER).
- **Metadados de pipeline** — `trace` com versão, duração, modo, passos e uso de API externa quando houver.
- **Comparação de versões de contrato** — endpoint dedicado `POST /analyze/compare` para diff assistido (fluxo separado da análise “única”).

**Modos (`fast` | `standard` | `deep`):** equilíbrio entre latência, uso de Qdrant/T5 e política de chamada ao LLM externo.

---

## API HTTP (superfície de integração)

| Método | Caminho | Uso |
|--------|---------|-----|
| `GET` | `/health` | Readiness: serviço e modelos carregados. |
| `POST` | `/analyze/file` | Multipart: arquivo + `regiao`, `mode`, `jobId`, `contractId`, etc. |
| `POST` | `/analyze/text` | JSON com texto já disponível (mesmo contrato de saída). |
| `POST` | `/analyze/file/async` | Enfileira análise; resposta **202** com `jobId` e `pollUrl`. |
| `GET` | `/jobs/{jobId}` | Polling: `queued` → `processing` → `done` ou `error`. |
| `POST` | `/analyze/compare` | Compara duas versões de contrato (schema próprio). |

A fila assíncrona usa **RabbitMQ**: a API publica jobs e o worker dedicado consome, processa e publica resultados.

---

## Ideias de produto (IA e dados) — maior parte no NestJS + este serviço como motor

Estas funcionalidades **ampliam o valor do escritório**; várias exigem apenas **novos endpoints ou jobs** que consomem o mesmo pipeline e cruzam com dados já guardados no PostgreSQL.

1. **Painel de performance por advogado** — taxa de uso da IA, tempo médio até envio da peça, tipos de caso mais analisados; correlation opcional com outcomes quando o escritório registrar resultado (privado).
2. **Benchmark interno do escritório** — distribuição de `risk.level`, quantidade de alertas críticos por área, comparado à média da própria base (anonimizada).
3. **Priorização inteligente da fila** — NestJS ordena jobs por urgência (audiência próxima, cliente estratégico) usando metadados; a IA só executa `standard`/`deep` nos prioritários.
4. **“Segunda leitura” automática** — reexecutar análise quando o modelo ou o índice Qdrant atualizarem; notificar diferenças relevantes (`trace.pipelineVersion`).
5. **Sugestão de cláusulas e checklist** — pós-processamento com LLM restrito a modelo de contrato do escritório + lista de compliance (LGPD, CDC, etc.).
6. **Simulação de cenários** — “e se o fundamento fosse X?”: segunda chamada com texto editado pelo NestJS (`/analyze/text`) sem persistir no mesmo `contractId` de produção.
7. **Alertas proativos** — workflows que escaneiam contratos em pasta (`draft`) e disparam webhook quando `risk` ou severidade cruzam limiar.
8. **Explicabilidade para o cliente final** — usar `attentionPoints` + `similarCases` resumidos em linguagem acessível (camada no NestJS ou LLM com política de segurança).
9. **Detecção de inconsistências** — comparar valores/datas entre `entities` e tabela de valores do processo (NestJS cruza com dados já estruturados).
10. **Telemetria de custo** — agregar `trace.estimatedCostUsd` e tokens por escritório para governança de uso do LLM externo.

---

## Stack técnica (resumo)

- **FastAPI**, **PyTorch / HuggingFace**, **Qdrant**, extração PDF/DOCX (com OCR quando necessário).
- Pesos em `hf_models/` (não versionar binários pesados no Git).
- Notebooks em `notebooks/` para retreino; scripts em `scripts/` para ingestão no Qdrant.

---

## Como rodar

```bash
docker compose up --build

# Popular Qdrant na primeira vez (exemplo)
pip install qdrant-client
python scripts/import_qdrant.py casos_com_embeddings.json

curl http://localhost:8000/health
curl -X POST http://localhost:8000/analyze/file \
  -F "file=@scripts/processo_teste.txt" -F "regiao=SP"
```

Testes: `./.venv/bin/python -m pytest api/ml/tests tests`

---

## Variáveis de ambiente relevantes

| Variável | Padrão | Função |
|----------|--------|--------|
| `MODELS_DIR` | `./hf_models` | Diretório dos pesos. |
| `QDRANT_HOST` / `QDRANT_PORT` | `localhost` / `6333` | Conexão ao vetorial. |
| `MAX_UPLOAD_BYTES` | `52428800` | Limite de upload (~50 MB). |
| `CORS_ORIGINS` | `*` | CORS (ajuste em produção). |
| `JURISPRO_SKIP_PRELOAD` | — | Não pré-carregar todos os modelos (CI/dev). |
| `JURISPRO_DEBUG_ERRORS` | — | Incluir `detail` interno em erros (**não** em produção). |
| `JURISPRO_OPENAI_*` | — | LLM compatível com OpenAI para enriquecimento opcional. |
| `JURISPRO_MAX_ASYNC_JOBS` | `500` | Máximo de metadados de jobs mantidos para polling local. |

---

## Limitações e uso responsável

- Modelos são **aproximações** treinadas em corpora limitados; podem errar em textos híbridos ou muito longos.
- **Casos semelhantes** refletem proximidade semântica na base indexada, não garantia de aplicabilidade ao caso concreto.
- **Probabilidade de desfecho** é **apoio à decisão**, não previsão judicial.
- **Honorários** são estimativas automáticas; o advogado deve validar contra tabela local e política do escritório.

Para backlog técnico detalhado, veja `MELHORIAS.md` e `tasks/`.
