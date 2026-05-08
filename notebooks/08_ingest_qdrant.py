# %% [markdown]
# # 08 — Geração de Embeddings para Qdrant
#
# Roda no Google Colab (GPU).
# Saída: `casos_com_embeddings.json` — download automático ao final.

# %%
!pip install sentence-transformers datasets requests -q

# %% [markdown]
# ## 1. GPU check

# %%
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# %% [markdown]
# ## 2. Imports

# %%
import json
import uuid
import requests
import numpy as np
from collections import Counter
from datasets import load_dataset, DatasetDict
from sentence_transformers import SentenceTransformer
from google.colab import files

# %% [markdown]
# ## 3. Carregar dataset

# %%
def load_dataset_robust(name: str) -> DatasetDict:
    try:
        return load_dataset(name)
    except RuntimeError as e:
        if "no longer supported" not in str(e):
            raise
        print(f"[fallback] carregando parquet do Hub: {name}")
        resp = requests.get(
            "https://datasets-server.huggingface.co/parquet",
            params={"dataset": name},
            timeout=30,
        )
        resp.raise_for_status()
        files_map: dict[str, list[str]] = {}
        for item in resp.json().get("parquet_files", []):
            files_map.setdefault(item["split"], []).append(item["url"])
        if not files_map:
            raise RuntimeError(f"Sem parquet disponível para '{name}'")
        return load_dataset("parquet", data_files=files_map)

print("Carregando dataset...")
ds = load_dataset_robust("joelniklaus/brazilian_court_decisions")
print(ds)

# %% [markdown]
# ## 4. Normalizar registros

# %%
OUTCOME_MAP = {
    "yes":         "procedente",
    "no":          "improcedente",
    "partial":     "parcialmente procedente",
    "unanalysted": "não analisado",
}

def infer_tipo(texto: str) -> str:
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

def build_records(ds: DatasetDict) -> list[dict]:
    records = []
    for split in ds.keys():
        for row in ds[split]:
            ementa = (row.get("decision_description") or "").strip()
            if len(ementa) < 50:
                continue
            outcome_raw = (row.get("judgment_label") or row.get("label") or "").lower()
            outcome = OUTCOME_MAP.get(outcome_raw, outcome_raw or "desconhecido")
            records.append({
                "id":       str(uuid.uuid4()),
                "tribunal": str(row.get("court") or ""),
                "tipo":     infer_tipo(ementa),
                "outcome":  outcome,
                "titulo":   ementa[:120].rstrip() + ("…" if len(ementa) > 120 else ""),
                "resumo":   ementa,
            })
    return records

records = build_records(ds)
print(f"\nTotal de registros: {len(records)}")
print("\nDistribuição por tipo:")
for tipo, qtd in Counter(r["tipo"] for r in records).most_common():
    print(f"  {tipo:<20} {qtd}")
print("\nDistribuição por outcome:")
for outcome, qtd in Counter(r["outcome"] for r in records).most_common():
    print(f"  {outcome:<30} {qtd}")

# %% [markdown]
# ## 5. Carregar modelo de embeddings
#
# Faça upload do arquivo `embeddings.zip` (gerado no notebook 05)
# antes de executar esta célula.

# %%
import os, shutil, zipfile

print("Faça upload do embeddings.zip gerado no notebook 05:")
uploaded = files.upload()

zip_name = list(uploaded.keys())[0]
extract_dir = "./hf_models/embeddings"
os.makedirs(extract_dir, exist_ok=True)

with zipfile.ZipFile(zip_name, "r") as z:
    z.extractall(extract_dir)

print(f"Modelo extraído em: {extract_dir}")
print("Conteúdo:", os.listdir(extract_dir))

# %%
model = SentenceTransformer(extract_dir, device=device)
print(f"Modelo carregado. Dimensão: {model.get_sentence_embedding_dimension()}")

# %% [markdown]
# ## 6. Gerar embeddings em batch (GPU)

# %%
BATCH_SIZE = 128  # aumentar se tiver GPU com mais VRAM

texts = [r["resumo"] for r in records]
print(f"Gerando embeddings para {len(texts)} textos (batch={BATCH_SIZE})...")

with torch.inference_mode():
    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
        device=device,
    )

print(f"Shape dos embeddings: {embeddings.shape}")

# %% [markdown]
# ## 7. Montar JSON final e fazer download

# %%
output = []
for record, emb in zip(records, embeddings):
    output.append({
        **record,
        "embedding": emb.tolist(),
    })

OUTPUT_FILE = "casos_com_embeddings.json"
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False)

size_mb = os.path.getsize(OUTPUT_FILE) / (1024 * 1024)
print(f"Arquivo gerado: {OUTPUT_FILE} ({size_mb:.1f} MB)")
print(f"Total de casos: {len(output)}")

files.download(OUTPUT_FILE)
print("Download iniciado!")
