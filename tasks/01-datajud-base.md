# 01 - Base DataJud e Qdrant

## Objetivo

Substituir a base atual de `30400` casos por uma base maior treinada/exportada com `17000` por tribunal comercial, mirando aproximadamente `300k` casos.

## Entrada esperada

- ZIP novo com:
  - `casos_datajud.json`
  - `manifest.json`
- Embeddings já gerados.
- Dimensão esperada: `768`.

## Tasks

- Validar se o ZIP abre corretamente.
- Validar `manifest.json`.
- Confirmar `max_por_tribunal`.
- Confirmar `total_records`.
- Confirmar contagem por tribunal.
- Confirmar que nenhum tribunal passou do limite configurado.
- Confirmar que todos os registros têm `embedding`.
- Confirmar dimensão uniforme dos embeddings.
- Confirmar IDs únicos.
- Confirmar `numeroProcesso` único quando existir.
- Importar no Qdrant com `--reset`.
- Validar `points_count` no Qdrant.
- Fazer consulta amostral e confirmar `data_source=datajud`.
- Testar `/analyze/file` e confirmar `similar_cases`.

## Comando de importação

```bash
.venv/bin/python scripts/import_qdrant.py CAMINHO_DO_ZIP_OU_JSON --reset
```

## Critério de pronto

- Qdrant contém somente a nova base DataJud.
- `points_count` bate com o total validado.
- `/analyze/file` retorna casos semelhantes.
- Tempo de resposta medido com a base maior.
