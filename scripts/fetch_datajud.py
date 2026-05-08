"""
Busca acórdãos públicos do DataJud (CNJ) e ingere no Qdrant.

API gratuita — requer cadastro em: https://datajud-wiki.cnj.jus.br
Defina a variável DATAJUD_API_KEY antes de rodar.

Uso:
    export DATAJUD_API_KEY=sua_chave
    python scripts/fetch_datajud.py
    python scripts/fetch_datajud.py --tribunal TJSP --max 5000
    python scripts/fetch_datajud.py --reset   # recria collection
"""

from __future__ import annotations

import argparse
import os
import time
import uuid
from pathlib import Path

import requests
import torch
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer

COLLECTION    = "casos_juridicos"
EMBEDDING_DIM = 768
BATCH_SIZE    = 64
MODEL_PATH    = str(Path(__file__).resolve().parent.parent / "hf_models" / "embeddings")

# Tribunais disponíveis no DataJud
TRIBUNAIS = [
    "TJSP", "TJRJ", "TJMG", "TJRS", "TJPR", "TJSC", "TJBA",
    "TJGO", "TJPE", "TJCE", "TJDF", "TJPA", "TJMA", "TJPI",
    "TRT2", "TRT3", "TRT4", "TRT15",
    "STJ", "STF", "TRF1", "TRF3", "TRF4",
]

DATAJUD_BASE = "https://api-publica.datajud.cnj.jus.br"

OUTCOME_MAP = {
    "procedente":              "procedente",
    "improcedente":            "improcedente",
    "parcialmente procedente": "parcialmente procedente",
    "provido":                 "procedente",
    "não provido":             "improcedente",
    "desprovido":              "improcedente",
}

TIPO_MAP = {
    "trabalhista":    "Trabalhista",
    "consumidor":     "Consumidor",
    "tributário":     "Tributário",
    "tributario":     "Tributário",
    "previdenciário": "Previdenciário",
    "previdenciario": "Previdenciário",
    "criminal":       "Criminal",
    "penal":          "Criminal",
    "família":        "Família",
    "familia":        "Família",
    "tecnologia":     "Tecnologia",
}


def infer_tipo(texto: str) -> str:
    t = texto.lower()
    if any(w in t for w in ["trabalhista", "clt", "horas extras", "fgts", "reclamante"]):
        return "Trabalhista"
    if any(w in t for w in ["consumidor", "cdc", "negativação", "plano de saúde", "operadora"]):
        return "Consumidor"
    if any(w in t for w in ["tributário", "icms", "iss", "imposto", "receita federal"]):
        return "Tributário"
    if any(w in t for w in ["previdenciário", "inss", "aposentadoria", "benefício"]):
        return "Previdenciário"
    if any(w in t for w in ["criminal", "penal", "réu", "condenação criminal"]):
        return "Criminal"
    if any(w in t for w in ["família", "divórcio", "alimentos", "guarda"]):
        return "Família"
    return "Outros"


def fetch_acordaos(
    tribunal: str,
    api_key: str,
    max_results: int = 1000,
    size: int = 100,
) -> list[dict]:
    headers = {"Authorization": f"APIKey {api_key}"}
    endpoint = f"{DATAJUD_BASE}/api_publica_{tribunal.lower()}/_search"

    body = {
        "size": size,
        "query": {
            "bool": {
                "must": [
                    {"match": {"tipoDocumento": "acórdão"}},
                ]
            }
        },
        "_source": ["ementa", "orgaoJulgador", "dataJulgamento", "movimentos"],
        "sort": [{"dataJulgamento": {"order": "desc"}}],
    }

    records = []
    fetched = 0
    search_after = None

    while fetched < max_results:
        if search_after:
            body["search_after"] = search_after

        try:
            resp = requests.post(endpoint, json=body, headers=headers, timeout=30)
            resp.raise_for_status()
        except requests.HTTPError as e:
            print(f"  HTTP {e.response.status_code} em {tribunal}: {e}")
            break
        except Exception as e:
            print(f"  Erro ao buscar {tribunal}: {e}")
            break

        hits = resp.json().get("hits", {}).get("hits", [])
        if not hits:
            break

        for h in hits:
            src = h.get("_source", {})
            ementa = (src.get("ementa") or "").strip()
            if len(ementa) < 50:
                continue

            # Tenta extrair outcome dos movimentos
            outcome = "desconhecido"
            for mov in (src.get("movimentos") or []):
                nome = (mov.get("nome") or "").lower()
                for k, v in OUTCOME_MAP.items():
                    if k in nome:
                        outcome = v
                        break
                if outcome != "desconhecido":
                    break

            records.append({
                "id":       str(uuid.uuid4()),
                "tribunal": tribunal,
                "tipo":     infer_tipo(ementa),
                "outcome":  outcome,
                "titulo":   ementa[:120] + ("…" if len(ementa) > 120 else ""),
                "resumo":   ementa,
            })

        fetched += len(hits)
        search_after = hits[-1].get("sort")
        print(f"  {tribunal}: {fetched} acórdãos coletados", end="\r")
        time.sleep(0.3)  # respeitar rate limit

    print()
    return records


def encode_batch(model: SentenceTransformer, texts: list[str]) -> list[list[float]]:
    all_embs = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        with torch.inference_mode():
            embs = model.encode(batch, normalize_embeddings=True, show_progress_bar=False)
        all_embs.extend(embs.tolist())
        print(f"  Embeddings: {min(i + BATCH_SIZE, len(texts))}/{len(texts)}", end="\r")
    print()
    return all_embs


def upsert(client: QdrantClient, records: list[dict], embeddings: list[list[float]]) -> None:
    for i in range(0, len(records), 256):
        batch_r = records[i : i + 256]
        batch_e = embeddings[i : i + 256]
        points = [
            PointStruct(
                id=r["id"],
                vector=e,
                payload={k: r[k] for k in ("tribunal", "tipo", "outcome", "titulo", "resumo")},
            )
            for r, e in zip(batch_r, batch_e)
        ]
        client.upsert(collection_name=COLLECTION, points=points)
        print(f"  Upsert: {min(i + 256, len(records))}/{len(records)}", end="\r")
    print()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tribunal", default=None, help=f"Ex: TJSP. Padrão: todos ({len(TRIBUNAIS)})")
    parser.add_argument("--max",      type=int, default=1000, help="Máximo por tribunal")
    parser.add_argument("--host",     default=os.getenv("QDRANT_HOST", "localhost"))
    parser.add_argument("--port",     type=int, default=int(os.getenv("QDRANT_PORT", "6333")))
    parser.add_argument("--reset",    action="store_true")
    args = parser.parse_args()

    api_key = os.getenv("DATAJUD_API_KEY", "")
    if not api_key:
        print("ERRO: defina a variável DATAJUD_API_KEY")
        print("Cadastro gratuito: https://datajud-wiki.cnj.jus.br")
        return

    tribunais = [args.tribunal] if args.tribunal else TRIBUNAIS

    print(f"Conectando ao Qdrant em {args.host}:{args.port}...")
    client = QdrantClient(host=args.host, port=args.port)

    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION in existing and args.reset:
        client.delete_collection(COLLECTION)
        print(f"Collection '{COLLECTION}' removida.")
    if COLLECTION not in [c.name for c in client.get_collections().collections]:
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        )
        print(f"Collection '{COLLECTION}' criada.")

    print(f"\nCarregando modelo: {MODEL_PATH}")
    model = SentenceTransformer(MODEL_PATH)

    total_inserido = 0
    for tribunal in tribunais:
        print(f"\n── {tribunal} ──")
        records = fetch_acordaos(tribunal, api_key, max_results=args.max)
        if not records:
            print(f"  Nenhum acórdão válido.")
            continue

        texts = [r["resumo"] for r in records]
        embeddings = encode_batch(model, texts)
        upsert(client, records, embeddings)
        total_inserido += len(records)
        print(f"  {len(records)} acórdãos inseridos.")

    info = client.get_collection(COLLECTION)
    print(f"\nConcluído! Total na collection: {info.points_count} pontos (+{total_inserido} novos).")


if __name__ == "__main__":
    main()
