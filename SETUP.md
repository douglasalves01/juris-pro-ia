# Setup do JurisPro IA

Guia rapido para subir o microservico de IA localmente com Docker Compose.

## Pre-requisitos

Tenha instalado:

- Docker
- Docker Compose
- Git

## 1. Configure o `.env`

Na raiz do projeto, copie o arquivo de exemplo:

```bash
cp .env.example .env
```

Para rodar via Docker Compose, mantenha estes valores:

```env
QDRANT_HOST=qdrant
QDRANT_PORT=6333
MODELS_DIR=/app/hf_models
```

Se quiser usar Gemini, preencha no `.env`:

```env
JURISPRO_GEMINI_API_KEY=sua_chave
JURISPRO_GEMINI_MODEL=gemini-2.5-flash
```

## 2. Confira os modelos

O compose monta os modelos locais a partir da pasta:

```text
hf_models/
```

Se voce recebeu `hf_models.zip`, extraia antes de subir:

```bash
unzip hf_models.zip
```

## 3. Crie rede e volume externos

O `docker-compose.yml` usa uma rede e um volume externos. Crie uma vez:

```bash
docker network create jurispro-net
docker volume create juris-pro-api_jurispro_contracts
```

Se eles ja existirem, o Docker pode avisar que ja foram criados. Nesse caso, siga normalmente.

## 4. Suba o projeto

Para subir tudo:

```bash
docker compose up --build
```

A API fica em:

```text
http://localhost:8000
```

Documentacao Swagger:

```text
http://localhost:8000/docs
```

Healthcheck:

```bash
curl http://localhost:8000/health
```

## 5. Se nao tiver RabbitMQ rodando

O worker usa RabbitMQ do backend pela rede `jurispro-net`. Se voce quiser testar apenas a API de IA sem o worker, suba somente API e Qdrant:

```bash
docker compose up --build qdrant api
```

## 6. Popular o Qdrant

Existem dois jeitos de popular a collection `casos_juridicos`.

### Opcao A: importar arquivo ja pronto

Use quando voce ja tem `casos_com_embeddings.json` ou um `.zip` com esse JSON. Esse arquivo ja vem com os embeddings calculados, entao a importacao e mais direta.

```bash
docker compose exec api python scripts/import_qdrant.py casos_com_embeddings.json --host qdrant --port 6333
```

Para recriar a collection do zero:

```bash
docker compose exec api python scripts/import_qdrant.py casos_com_embeddings.json --host qdrant --port 6333 --reset
```

### Opcao B: baixar dataset publico e gerar embeddings

Use quando voce quer popular direto a partir do dataset publico do Hugging Face. Esse caminho baixa `joelniklaus/brazilian_court_decisions`, gera embeddings com o modelo local em `hf_models/embeddings` e grava tudo no Qdrant.

```bash
docker compose exec api python scripts/ingest_casos.py --host qdrant --port 6333
```

Para recriar a collection do zero:

```bash
docker compose exec api python scripts/ingest_casos.py --host qdrant --port 6333 --reset
```

Resumo:

- `scripts/import_qdrant.py`: importa um arquivo pronto com embeddings.
- `scripts/ingest_casos.py`: baixa dataset publico e gera embeddings usando `hf_models/embeddings`.

## 7. Testar a API

Analise de texto:

```bash
curl -X POST http://localhost:8000/analyze/text \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Contrato de prestacao de servicos com multa de 50% por rescisao antecipada.",
    "regiao": "SP",
    "mode": "standard"
  }'
```

Analise de arquivo:

```bash
curl -X POST http://localhost:8000/analyze/file \
  -F "file=@fixtures/processo_sintetico.pdf" \
  -F "regiao=SP" \
  -F "mode=standard"
```

## Comandos uteis

Ver logs:

```bash
docker compose logs -f
```

Ver logs so da API:

```bash
docker compose logs -f api
```

Parar:

```bash
docker compose down
```

Parar e remover volumes criados pelo compose:

```bash
docker compose down -v
```

Rodar testes dentro do container:

```bash
docker compose exec api python -m pytest
```

## Problemas comuns

### Rede externa nao encontrada

```bash
docker network create jurispro-net
```

### Volume externo nao encontrado

```bash
docker volume create juris-pro-api_jurispro_contracts
```

### Worker reiniciando

Provavelmente o RabbitMQ do backend nao esta rodando. Para desenvolver so a API, use:

```bash
docker compose up --build qdrant api
```

### Modelos nao encontrados

Confira se a pasta `hf_models/` existe na raiz do projeto. O compose monta essa pasta em `/app/hf_models`.
