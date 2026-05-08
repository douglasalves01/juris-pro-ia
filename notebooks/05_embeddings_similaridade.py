
!pip install sentence-transformers datasets accelerate requests -q

# %% [markdown]
# ## 2. Imports

# %%
import torch
import numpy as np
import random
import requests
from datasets import load_dataset, DatasetDict
from sentence_transformers import (
    SentenceTransformer,
    InputExample,
    losses,
    evaluation,
    util,
)
from torch.utils.data import DataLoader

random.seed(42)
np.random.seed(42)

# %%
# ── HELPER: fallback para parquet do Hub ────────────────────────────────────

def load_dataset_robust(name: str, **kwargs) -> DatasetDict:
    try:
        return load_dataset(name, **kwargs)
    except RuntimeError as e:
        if "no longer supported" not in str(e):
            raise
        print(f"[fallback] loading script bloqueado — carregando parquet do Hub: {name}")
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
            raise RuntimeError(f"Sem parquet disponível para '{name}'. Tente: pip install 'datasets<4.0'")
        return load_dataset("parquet", data_files=files)

# %% [markdown]
# ## 3. Carregar datasets

# %%
# Dataset de sentenças jurídicas PT para similaridade
legal_sents = load_dataset_robust("stjiris/portuguese-legal-sentences-v0")
print("Legal sentences:", legal_sents)

# Ementas para ampliar o corpus
court_dec = load_dataset_robust("joelniklaus/brazilian_court_decisions")
print("\nCourt decisions:", court_dec)

# %% [markdown]
# ## 4. Preparar pares de treinamento
#
# Para treinar com MultipleNegativesRankingLoss precisamos de pares
# (anchor, positive) — duas sentenças com significado similar.
# Estratégia:
#   a) Sentenças do mesmo documento (vizinhas) → positive
#   b) Sentenças de documentos distintos       → negative (automático pelo loss)

# %%
def build_sts_pairs_from_sentences(dataset, split="train", max_pairs=5000):
    """
    Cria pares de sentenças jurídicas do mesmo contexto como exemplos positivos.
    """
    sents = dataset[split]["sentence"] if "sentence" in dataset[split].column_names \
            else dataset[split]["text"]

    pairs = []
    sents = [s.strip() for s in sents if s and len(s.strip()) > 30]

    for i in range(0, min(len(sents) - 1, max_pairs * 2), 2):
        a, b = sents[i], sents[i + 1]
        if len(a) > 20 and len(b) > 20:
            pairs.append(InputExample(texts=[a, b]))
        if len(pairs) >= max_pairs:
            break

    return pairs

def build_pairs_from_ementas(dataset, split="train", max_pairs=3000):
    """
    Usa fragmentos da mesma ementa como pares positivos.
    """
    ementas = [e.strip() for e in dataset[split]["decision_description"]
               if e and len(e.strip()) > 80]

    pairs = []
    for ementa in ementas[:max_pairs]:
        words = ementa.split()
        mid = len(words) // 2
        if mid > 10:
            a = " ".join(words[:mid])
            b = " ".join(words[mid:])
            pairs.append(InputExample(texts=[a, b]))

    return pairs

# Construir pares
try:
    pairs_sents  = build_sts_pairs_from_sentences(legal_sents, max_pairs=4000)
    print(f"Pares de legal sentences: {len(pairs_sents)}")
except Exception as e:
    print(f"Aviso legal_sents: {e}")
    pairs_sents = []

pairs_ementas = build_pairs_from_ementas(court_dec, max_pairs=3000)
print(f"Pares de ementas: {len(pairs_ementas)}")

all_pairs = pairs_sents + pairs_ementas
random.shuffle(all_pairs)
print(f"Total de pares de treinamento: {len(all_pairs)}")

# Split
n_val   = max(50, int(len(all_pairs) * 0.1))
n_train = len(all_pairs) - n_val

train_pairs = all_pairs[:n_train]
val_pairs   = all_pairs[n_train:]
print(f"Treino: {len(train_pairs)} | Val: {len(val_pairs)}")

# %% [markdown]
# ## 5. Carregar modelo base (já treinado em dados jurídicos PT)

# %%
# rufimelo/Legal-BERTimbau-large-STSB é pré-treinado em STS jurídico PT-BR
# Fine-tuning aqui melhora ainda mais para o domínio específico
MODEL_NAME = "rufimelo/Legal-BERTimbau-large-STSB"

try:
    model = SentenceTransformer(MODEL_NAME)
    print(f"Modelo carregado: {MODEL_NAME}")
except Exception:
    # Fallback para modelo base menor se o large não caber na VRAM
    MODEL_NAME = "neuralmind/bert-base-portuguese-cased"
    from sentence_transformers import models
    word_embedding = models.Transformer(MODEL_NAME, max_seq_length=512)
    pooling        = models.Pooling(word_embedding.get_word_embedding_dimension(),
                                    pooling_mode_mean_tokens=True)
    model = SentenceTransformer(modules=[word_embedding, pooling])
    print(f"Fallback para: {MODEL_NAME}")

print("Dimensão do embedding:", model.get_sentence_embedding_dimension())

# %% [markdown]
# ## 6. Loss e avaliador

# %%
train_loader = DataLoader(train_pairs, shuffle=True, batch_size=16)
train_loss   = losses.MultipleNegativesRankingLoss(model)

# Avaliador de similaridade semântica
val_anchors   = [p.texts[0] for p in val_pairs]
val_positives = [p.texts[1] for p in val_pairs]

evaluator = evaluation.EmbeddingSimilarityEvaluator(
    sentences1=val_anchors,
    sentences2=val_positives,
    scores=[1.0] * len(val_pairs),  # todos positivos → similaridade 1.0
    name="val_sts",
)

# %% [markdown]
# ## 7. Fine-tuning

# %%
OUTPUT_DIR = "./embeddings_output"

model.fit(
    train_objectives=[(train_loader, train_loss)],
    evaluator=evaluator,
    epochs=3,
    warmup_steps=int(len(train_loader) * 0.1),
    evaluation_steps=200,
    output_path=OUTPUT_DIR,
    save_best_model=True,
    show_progress_bar=True,
    use_amp=torch.cuda.is_available(),
)

print("Treinamento concluído.")

# %% [markdown]
# ## 8. Salvar no Google Drive

# %%
import shutil
from google.colab import files

SAVE_DIR = "./hf_models/embeddings"
model.save(SAVE_DIR)

shutil.make_archive(SAVE_DIR, "zip", SAVE_DIR)
files.download("embeddings.zip")
print("Download iniciado: embeddings.zip")

# %% [markdown]
# ## 9. Construir índice de casos similares (simulação)
#
# Em produção, esses vetores ficam no pgvector.
# Aqui demonstramos a busca com numpy puro.

# %%
from sentence_transformers import SentenceTransformer, util

model_inf = SentenceTransformer(SAVE_DIR)

# Base de conhecimento simulada
BASE_CASOS = [
    {
        "id": "TJSP-001",
        "titulo": "Indenização por negativação indevida — Telecom",
        "resumo": "Consumidor negativado sem débito. Operadora não provou a dívida. "
                  "Condenação em R$ 8.000 por danos morais.",
        "outcome": "procedente",
        "tipo": "Consumidor",
    },
    {
        "id": "TRT2-045",
        "titulo": "Horas extras — Sistema de ponto eletrônico adulterado",
        "resumo": "Reclamante demonstrou irregularidade no controle de jornada. "
                  "Empresa condenada ao pagamento de 180h extras.",
        "outcome": "parcialmente procedente",
        "tipo": "Trabalhista",
    },
    {
        "id": "STJ-234",
        "titulo": "Revisão contratual — Juros abusivos banco digital",
        "resumo": "Taxa de juros superior ao dobro da média do mercado. "
                  "Contrato revisado com devolução de valores.",
        "outcome": "procedente",
        "tipo": "Consumidor",
    },
    {
        "id": "TJRJ-089",
        "titulo": "Rescisão de contrato de TI por descumprimento de SLA",
        "resumo": "Empresa de software descumpriu prazo de entrega e SLA mínimo. "
                  "Multa rescisória de 20% aplicada.",
        "outcome": "procedente",
        "tipo": "Tecnologia",
    },
    {
        "id": "TRF5-567",
        "titulo": "Aposentadoria por tempo de contribuição — reconhecimento rural",
        "resumo": "Segurado comprovou atividade rural com documentos e testemunhas. "
                  "Reconhecido período e concedida aposentadoria.",
        "outcome": "procedente",
        "tipo": "Previdenciário",
    },
    {
        "id": "TJMG-321",
        "titulo": "Plano de saúde — negativa de cobertura cirurgia ortopédica",
        "resumo": "Plano de saúde recusou procedimento com indicação médica. "
                  "Sentença obrigou cobertura + danos morais R$ 6.000.",
        "outcome": "procedente",
        "tipo": "Consumidor",
    },
]

# Gerar embeddings da base
base_textos = [c["resumo"] for c in BASE_CASOS]
base_embeds = model_inf.encode(base_textos, convert_to_tensor=True, show_progress_bar=False)

def buscar_casos_similares(texto_consulta: str, top_k: int = 3) -> list:
    """
    Retorna os top_k casos mais similares ao texto da consulta.
    Em produção: substitua por SELECT ... ORDER BY embedding <=> $1 LIMIT k
    usando pgvector no PostgreSQL.
    """
    embed_consulta = model_inf.encode(texto_consulta, convert_to_tensor=True)
    scores = util.cos_sim(embed_consulta, base_embeds)[0]
    top_indices = scores.argsort(descending=True)[:top_k]

    resultados = []
    for idx in top_indices:
        caso = BASE_CASOS[idx].copy()
        caso["similaridade"] = round(float(scores[idx]), 4)
        resultados.append(caso)

    return resultados

# Teste
consultas = [
    "Cliente com nome no SPC por débito que não reconhece. Operadora não apresentou contrato.",
    "Trabalhador sem registro de horas extras no sistema da empresa.",
    "Contrato de software com atraso de 8 meses na entrega.",
]

print("=== Busca de Casos Similares ===")
for consulta in consultas:
    print(f"\nConsulta: {consulta}")
    similares = buscar_casos_similares(consulta, top_k=2)
    for s in similares:
        print(f"  [{s['id']}] {s['titulo']}")
        print(f"  Tipo: {s['tipo']} | Outcome: {s['outcome']} | Similarity: {s['similaridade']:.3f}")
        print(f"  Resumo: {s['resumo'][:100]}...")
