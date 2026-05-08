# 06 - Parecer Final Estruturado

## Objetivo

Gerar `finalOpinion` estruturado para exibição e exportação.

## Estrutura

```json
{
  "title": "Parecer preliminar sobre documento consumerista",
  "executiveSummary": "Resumo objetivo.",
  "legalAnalysis": "Análise jurídica estruturada.",
  "recommendations": [],
  "limitations": []
}
```

## Fonte inicial

No MVP, pode ser montado com:

- `executive_summary`
- `main_risks`
- `recommendations`
- `positive_points`
- `similar_cases`

## Evolução

- No modo `deep`, permitir OpenAI/Gemini para melhorar argumentação.
- Separar fatos extraídos de inferências.
- Citar evidências usadas.

## Critério de pronto

- `finalOpinion` sempre existe no v2.
- No modo sem API externa, parecer é simples, mas estruturado.
- No modo deep, parecer pode ser enriquecido.
