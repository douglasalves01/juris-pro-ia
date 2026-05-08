# 07 - Worker e Fila

## Objetivo

Preparar a IA para integração assíncrona com backend NestJS.

## Opções

### Curto prazo

HTTP interno:

```http
POST /analyze/file/async
GET /jobs/{jobId}
```

### Produção

Fila:

- `ai.analysis.requested`
- `ai.analysis.progress`
- `ai.analysis.completed`
- `ai.analysis.failed`

## Tasks

- Definir payload de request com `jobId`, `firmId`, `contractId`.
- Criar modelo de progresso.
- Criar endpoint/status ou publicador de evento.
- Garantir que o worker consiga processar sem bloquear request HTTP.
- Definir cancelamento futuro.

## Critério de pronto

- Backend consegue acompanhar progresso.
- Falhas retornam evento/erro padronizado.
- Pipeline pode rodar em background.
