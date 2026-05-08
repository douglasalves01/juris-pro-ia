# 08 - Testes e Qualidade

## Objetivo

Garantir estabilidade do contrato e da qualidade básica da análise.

## Testes mínimos do endpoint atual

- `/health` retorna `200`.
- `/analyze/file` com TXT retorna `200`.
- `/analyze/file` com PDF retorna `200`.
- `/analyze/file` com DOCX retorna `200`.
- Arquivo inválido retorna erro HTTP com `detail`.
- Arquivo vazio retorna erro HTTP com `detail`.

## Contrato JSON (análise)

- Arquivo inválido retorna erro no contrato (`status: error`, `error.code`, `trace.steps`).
- Arquivo vazio / sem texto retorna erro no contrato.
- Resposta de sucesso valida contra `schemas/analysis-response.schema.json`.
- `risk.score` fica entre 0 e 100.
- `risk.level` bate com faixa do score.
- `similarCases` retorna casos DataJud quando Qdrant está populado.
- `trace.pipelineVersion` existe.
- `trace.steps` não vem vazio.
- `POST /analyze/file/async` + `GET /jobs/{jobId}`: fila e polling.

## Testes manuais úteis

```bash
curl -i http://127.0.0.1:8000/health
```

```bash
curl -i -X POST \
  -F 'file=@scripts/processo_teste.txt;type=text/plain' \
  -F regiao=SP \
  http://127.0.0.1:8000/analyze/file
```

## Critério de pronto

- Testes principais passam localmente.
- Contrato não quebra quando modelos retornam vazio/parcial.
- Erros não expõem stack trace para o backend.
