# 00 - Overview das Tasks para Fechar IA 100%

## Estado atual

- API FastAPI rodando com `POST /analyze/file`.
- Recebe PDF, DOCX, TXT e TEXT.
- Pipeline já extrai texto, classifica, calcula risco, entidades, honorários, probabilidade e casos semelhantes.
- Qdrant populado somente com DataJud.
- Base atual: `30400` casos.
- Endpoint testado com `scripts/processo_teste.txt` retornando `200 OK`.

## Objetivo

Transformar a IA atual em uma pipeline pronta para integração SaaS com backend NestJS, com contrato estável, rastreabilidade, erros padronizados, base vetorial forte e resposta auditável.

## Ordem recomendada

1. `01-datajud-base.md`
2. `02-contract-v2.md`
3. `03-error-contract.md`
4. `04-analysis-modes.md`
5. `05-trace-cost-versioning.md`
6. `06-final-opinion.md`
7. `07-worker-queue.md`
8. `08-tests-quality.md`

## Critério de pronto geral

- Backend consegue enviar documento e receber JSON validável.
- Resultado inclui `jobId`, `contractId`, `status`, `result` e `trace`.
- Erros seguem contrato padronizado.
- Base DataJud maior está importada e validada.
- Similar cases vêm do Qdrant.
- Pipeline informa versão, modelos, tempos e se usou API externa.
- Testes cobrem sucesso e falhas principais.
