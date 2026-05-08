# 02 - Contrato da API para o Backend (histórico)

> **Nota:** o contrato camelCase com `jobId`, `trace`, etc. está unificado em `POST /analyze/file`, `POST /analyze/text`, fila `POST /analyze/file/async` e `GET /jobs/{jobId}` — sem prefixo de versão na URL.

## Objetivo

Resposta compatível com o contrato esperado pelo backend NestJS.

## Endpoints

```http
POST /analyze/file
POST /analyze/text
POST /analyze/file/async
GET /jobs/{jobId}
```

## Entrada

`POST /analyze/file` e `/analyze/file/async` usam `multipart/form-data`:

- `file`: obrigatório.
- `jobId`: opcional no começo.
- `contractId`: opcional no começo.
- `firmId`: opcional no começo.
- `userId`: opcional.
- `mode`: `fast`, `standard` ou `deep`.
- `regiao`: default `SP`.

Para worker/fila em produção, o backend pode evoluir para enviar JSON com `file.path`, `file.url`, `mimeType`, `jobId`, `firmId` e `contractId`. Isso fica separado da rota HTTP de upload.

## Saída esperada

```json
{
  "jobId": "job-id",
  "contractId": "contract-id",
  "status": "done",
  "result": {
    "document": {},
    "risk": {},
    "attentionPoints": [],
    "entities": [],
    "similarCases": [
      {
        "caseId": "id",
        "tribunal": "TJSP",
        "number": "numeroProcesso",
        "similarity": 0.0,
        "outcome": "desconhecido",
        "summary": "resumo",
        "relevanceReason": "motivo"
      }
    ],
    "fees": {},
    "outcomeProbability": {},
    "finalOpinion": {}
  },
  "trace": {}
}
```

## Mapeamento da resposta atual

- `contract_type` -> `result.document.legalArea`
- `document_kind` -> `result.document.type`
- `executive_summary` -> `result.document.summary`
- `risk_score` -> `result.risk.score`
- `risk_level` -> `result.risk.level`
- `attention_points` -> `result.attentionPoints`
- `entities` -> `result.entities`
- `similar_cases` -> `result.similarCases`
- `fee_estimate_min/max/suggested` -> `result.fees`
- `win_probability` -> `result.outcomeProbability.value`
- `recommendations` + `executive_summary` -> `result.finalOpinion`

## Normalizações obrigatórias

- `risk_level` atual em português deve virar enum do backend:
  - `baixo` -> `BAIXO`
  - `médio`/`medio` -> `MEDIO`
  - `alto` -> `ALTO`
  - `crítico`/`critico` -> `CRITICO`
- `attention_points.severidade` atual deve virar:
  - `baixa` -> `low`
  - `média` -> `medium`
  - `alta` -> `high`
  - `crítica` -> `critical`
- `similar_cases` precisa preservar:
  - `tribunal`
  - `numeroProcesso` como `number`
  - `id` como `caseId`

## Critério de pronto

- As rotas de análise retornam sempre JSON no formato esperado.
- Campos obrigatórios existem mesmo quando vazios.
- Backend consegue validar schema sem tratamento especial.
