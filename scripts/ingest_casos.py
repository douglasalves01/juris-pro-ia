"""
Ingere jurisprudência pública no Qdrant.

Fontes (todas públicas, sem dados sensíveis):
  1. HuggingFace: joelniklaus/brazilian_court_decisions (~8k ementas)

Uso:
    python scripts/ingest_casos.py
    python scripts/ingest_casos.py --host localhost --port 6333
    python scripts/ingest_casos.py --reset   # recria a collection do zero
"""

from __future__ import annotations

import argparse
import os
import uuid
from pathlib import Path

import requests
import torch
from datasets import load_dataset, DatasetDict
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
)
from sentence_transformers import SentenceTransformer

# ── Configurações ────────────────────────────────────────────────────────────

COLLECTION     = "casos_juridicos"
EMBEDDING_DIM  = 768
BATCH_SIZE     = 64
MODEL_PATH     = str(Path(__file__).resolve().parent.parent / "hf_models" / "embeddings")

# Mapeamento outcome dataset → label legível
OUTCOME_MAP = {
    "yes":           "procedente",
    "no":            "improcedente",
    "partial":       "parcialmente procedente",
    "unanalysted":   "não analisado",
}

# ── Helpers ──────────────────────────────────────────────────────────────────

def load_dataset_robust(name: str) -> DatasetDict:
    try:
        return load_dataset(name)
    except RuntimeError as e:
        if "no longer supported" not in str(e):
            raise
        print(f"[fallback] Carregando parquet do Hub: {name}")
        resp = requests.get(
            "https://datasets-server.huggingface.co/parquet",
            params={"dataset": name},
            timeout=30,
        )
        resp.raise_for_status()
        files: dict[str, list[str]] = {}
        for item in resp.json().get("parquet_files", []):
            files.setdefault(item["split"], []).append(item["url"])
        if not files:
            raise RuntimeError(f"Sem parquet disponível para '{name}'")
        return load_dataset("parquet", data_files=files)


def build_records(ds: DatasetDict) -> list[dict]:
    """Normaliza o dataset para lista de dicts com campos padronizados."""
    records = []
    for split in ds.keys():
        for row in ds[split]:
            ementa = (row.get("decision_description") or "").strip()
            if len(ementa) < 50:
                continue

            outcome_raw = (row.get("judgment_label") or row.get("label") or "").lower()
            outcome = OUTCOME_MAP.get(outcome_raw, outcome_raw or "desconhecido")

            tipo = infer_tipo(ementa)

            records.append({
                "id":       str(uuid.uuid4()),
                "tribunal": str(row.get("court") or row.get("tribunal") or ""),
                "tipo":     tipo,
                "outcome":  outcome,
                "titulo":   ementa[:120].rstrip() + ("…" if len(ementa) > 120 else ""),
                "resumo":   ementa,
            })
    return records


def infer_tipo(texto: str) -> str:
    """Heurística simples para classificar tipo do caso pela ementa."""
    t = texto.lower()
    if any(w in t for w in ["trabalhista", "clt", "horas extras", "rescisão contratual", "fgts"]):
        return "Trabalhista"
    if any(w in t for w in ["consumidor", "cdc", "fornecedor", "negativação", "plano de saúde"]):
        return "Consumidor"
    if any(w in t for w in ["tributário", "icms", "iss", "imposto", "fisco", "receita federal"]):
        return "Tributário"
    if any(w in t for w in ["previdenciário", "inss", "aposentadoria", "benefício"]):
        return "Previdenciário"
    if any(w in t for w in ["criminal", "penal", "réu", "absolvição", "condenação criminal"]):
        return "Criminal"
    if any(w in t for w in ["família", "divórcio", "alimentos", "guarda", "inventário"]):
        return "Família"
    if any(w in t for w in ["software", "tecnologia", "ti ", "sistema", "licença de uso"]):
        return "Tecnologia"
    return "Outros"


def encode_in_batches(
    model: SentenceTransformer,
    texts: list[str],
    batch_size: int = BATCH_SIZE,
) -> list[list[float]]:
    all_embs: list[list[float]] = []
    total = len(texts)
    for i in range(0, total, batch_size):
        batch = texts[i : i + batch_size]
        with torch.inference_mode():
            embs = model.encode(
                batch,
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
        all_embs.extend(embs.tolist())
        print(f"  Embeddings: {min(i + batch_size, total)}/{total}", end="\r")
    print()
    return all_embs


def ensure_collection(client: QdrantClient, reset: bool) -> None:
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION in existing:
        if reset:
            client.delete_collection(COLLECTION)
            print(f"Collection '{COLLECTION}' removida (--reset).")
        else:
            print(f"Collection '{COLLECTION}' já existe — adicionando novos registros.")
            return
    client.create_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
    )
    print(f"Collection '{COLLECTION}' criada.")


def upsert_batch(
    client: QdrantClient,
    records: list[dict],
    embeddings: list[list[float]],
    batch_size: int = 256,
) -> None:
    total = len(records)
    for i in range(0, total, batch_size):
        batch_r = records[i : i + batch_size]
        batch_e = embeddings[i : i + batch_size]
        points = [
            PointStruct(
                id=r["id"],
                vector=e,
                payload={
                    "tribunal": r["tribunal"],
                    "tipo":     r["tipo"],
                    "outcome":  r["outcome"],
                    "titulo":   r["titulo"],
                    "resumo":   r["resumo"],
                },
            )
            for r, e in zip(batch_r, batch_e)
        ]
        client.upsert(collection_name=COLLECTION, points=points)
        print(f"  Upsert: {min(i + batch_size, total)}/{total}", end="\r")
    print()


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host",  default=os.getenv("QDRANT_HOST", "localhost"))
    parser.add_argument("--port",  type=int, default=int(os.getenv("QDRANT_PORT", "6333")))
    parser.add_argument("--reset", action="store_true", help="Recria a collection do zero")
    args = parser.parse_args()

    print(f"Conectando ao Qdrant em {args.host}:{args.port}...")
    client = QdrantClient(host=args.host, port=args.port)
    ensure_collection(client, reset=args.reset)

    # 1. Dataset HuggingFace
    print("\nCarregando dataset joelniklaus/brazilian_court_decisions...")
    ds = load_dataset_robust("joelniklaus/brazilian_court_decisions")
    records = build_records(ds)
    print(f"  Registros válidos: {len(records)}")

    if not records:
        print("Nenhum registro para ingerir. Abortando.")
        return

    # 3. Carregar modelo de embeddings
    print(f"\nCarregando modelo de embeddings: {MODEL_PATH}")
    if not Path(MODEL_PATH).is_dir():
        print(f"ERRO: modelo não encontrado em {MODEL_PATH}")
        print("Treine o notebook 05_embeddings_similaridade.py no Colab primeiro.")
        return
    model = SentenceTransformer(MODEL_PATH)
    print(f"  Dimensão: {model.get_sentence_embedding_dimension()}")

    # 4. Gerar embeddings
    print(f"\nGerando embeddings para {len(records)} casos...")
    texts = [r["resumo"] for r in records]
    embeddings = encode_in_batches(model, texts)

    # 5. Inserir no Qdrant
    print(f"\nInserindo no Qdrant...")
    upsert_batch(client, records, embeddings)

    info = client.get_collection(COLLECTION)
    print(f"\nConcluído! Total de pontos na collection: {info.points_count}")
    print(f"Distribuição por tipo:")
    from collections import Counter
    for tipo, qtd in Counter(r["tipo"] for r in records).most_common():
        print(f"  {tipo:<20} {qtd}")


if __name__ == "__main__":
    main()
