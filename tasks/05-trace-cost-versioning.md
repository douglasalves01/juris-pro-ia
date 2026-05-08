# 05 - Trace, Custo e Versionamento

## Objetivo

Tornar cada análise auditável e reprocessável.

## Trace esperado

```json
{
  "pipelineVersion": "2.0.0",
  "startedAt": "2026-04-19T18:00:00Z",
  "finishedAt": "2026-04-19T18:00:05Z",
  "durationMs": 5255,
  "externalApiUsed": false,
  "externalProvider": null,
  "externalModel": null,
  "steps": [
    {
      "step": "extract_text",
      "provider": "internal",
      "durationMs": 100,
      "confidence": 1.0
    }
  ]
}
```

## Tasks

- Definir `pipelineVersion`.
- Medir duração por etapa.
- Registrar modelos usados.
- Registrar versão dos modelos quando disponível.
- Registrar se API externa foi usada.
- Registrar tokens e custo quando houver API externa.
- Registrar custo local estimado como `0` ou valor configurável.

## Critério de pronto

- Toda resposta v2 tem `trace`.
- `trace.steps` lista etapas principais.
- Auditoria consegue saber qual pipeline/modelo gerou a análise.
