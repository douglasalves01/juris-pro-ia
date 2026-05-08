"""
Importa casos_com_embeddings.json (gerado no Colab) para o Qdrant local.

Uso:
    python scripts/import_qdrant.py casos_com_embeddings.json
    python scripts/import_qdrant.py casos_com_embeddings.json --reset
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import zipfile
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

COLLECTION    = "casos_juridicos"
BATCH_SIZE    = 256


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("arquivo", help="Path para casos_com_embeddings.json")
    parser.add_argument("--host",  default=os.getenv("QDRANT_HOST", "localhost"))
    parser.add_argument("--port",  type=int, default=int(os.getenv("QDRANT_PORT", "6333")))
    parser.add_argument("--reset", action="store_true", help="Recria a collection do zero")
    args = parser.parse_args()

    # Carregar JSON
    path = Path(args.arquivo)
    if not path.is_file():
        print(f"ERRO: arquivo não encontrado: {path}")
        sys.exit(1)

    print(f"Carregando {path}...")
    casos = load_cases(path)
    print(f"  {len(casos)} casos carregados.")

    if not casos:
        print("Nenhum caso no arquivo. Abortando.")
        sys.exit(1)

    # Detectar dimensão do embedding
    dim = len(casos[0]["embedding"])
    print(f"  Dimensão dos embeddings: {dim}")

    # Conectar ao Qdrant
    print(f"\nConectando ao Qdrant em {args.host}:{args.port}...")
    client = QdrantClient(host=args.host, port=args.port)

    # Gerenciar collection
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION in existing:
        if args.reset:
            client.delete_collection(COLLECTION)
            print(f"Collection '{COLLECTION}' removida (--reset).")
        else:
            info = client.get_collection(COLLECTION)
            print(f"Collection '{COLLECTION}' já existe com {info.points_count} pontos.")
            print("Use --reset para recriar do zero ou continue para adicionar.")

    if COLLECTION not in [c.name for c in client.get_collections().collections]:
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )
        print(f"Collection '{COLLECTION}' criada (dim={dim}).")

    # Inserir em batches
    total = len(casos)
    print(f"\nInserindo {total} casos no Qdrant...")

    for i in range(0, total, BATCH_SIZE):
        batch = casos[i : i + BATCH_SIZE]
        points = [
            PointStruct(
                id=caso["id"],
                vector=caso["embedding"],
                payload={k: v for k, v in caso.items() if k not in {"id", "embedding"}},
            )
            for caso in batch
        ]
        client.upsert(collection_name=COLLECTION, points=points)
        print(f"  {min(i + BATCH_SIZE, total)}/{total}", end="\r")

    print()
    info = client.get_collection(COLLECTION)
    print(f"\nConcluído! Total na collection: {info.points_count} pontos.")


def load_cases(path: Path) -> list[dict]:
    if path.suffix.lower() != ".zip":
        with path.open(encoding="utf-8") as f:
            return json.load(f)

    with zipfile.ZipFile(path) as zf:
        candidates = [
            name
            for name in zf.namelist()
            if name.endswith(".json") and not name.endswith("manifest.json")
        ]
        if not candidates:
            raise SystemExit("ERRO: zip sem arquivo JSON de casos.")
        with zf.open(candidates[0]) as f:
            return json.load(f)


if __name__ == "__main__":
    main()
