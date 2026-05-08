# 04 - Modos de Análise

## Objetivo

Suportar modos `fast`, `standard` e `deep` para controlar custo, tempo e uso de API externa.

## Modos

### fast

- Usar somente modelos locais e regras.
- Não chamar OpenAI/Gemini.
- Retornar análise rápida.

### standard

- Usar modelos locais.
- Chamar API externa apenas por gatilho.
- Gatilhos:
  - risco preliminar >= 70;
  - confiança baixa;
  - divergência entre modelos;
  - documento complexo;
  - muitos achados sem hierarquia.

### deep

- Permitir API externa para parecer final.
- Usar contexto maior.
- Melhor para documentos complexos ou reanálise profunda.

## Tasks

- Adicionar parâmetro `mode`.
- Criar configuração dos gatilhos.
- Retornar modo usado no `trace`.
- Bloquear API externa no modo `fast`.
- Preparar pontos de integração OpenAI/Gemini sem obrigar uso imediato.

## Critério de pronto

- Endpoint aceita `mode`.
- `fast` nunca usa API externa.
- `standard` só usa externa por gatilho.
- `deep` permite parecer final externo.
