# 03 - Contrato de Erros

## Objetivo

Padronizar erros para o backend decidir entre reprocessar, marcar falha ou pedir ação do usuário.

## Formato esperado

```json
{
  "jobId": "job-id",
  "contractId": "contract-id",
  "status": "error",
  "error": {
    "code": "UNSUPPORTED_FILE",
    "message": "Formato não suportado.",
    "retryable": false,
    "detail": {}
  },
  "trace": {}
}
```

## Códigos mínimos

- `UNSUPPORTED_FILE`
- `TEXT_EXTRACTION_FAILED`
- `OCR_FAILED`
- `MODEL_UNAVAILABLE`
- `EXTERNAL_API_FAILED`
- `OUTPUT_VALIDATION_FAILED`
- `DOCUMENT_TOO_LARGE`
- `TIMEOUT`
- `UNKNOWN`

## Tasks

- Criar helper para converter exceções em erro padronizado.
- Aplicar no endpoint v2.
- Manter status HTTP coerente.
- Garantir que erros também tenham `jobId`, `contractId` e `trace`.
- Testar arquivo inválido.
- Testar arquivo vazio.
- Testar timeout ou erro simulado de modelo.

## Critério de pronto

- Nenhum erro do endpoint v2 retorna stack trace crua.
- Todo erro é validável pelo backend.
