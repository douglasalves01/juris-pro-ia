# ============================================================
# NOTEBOOK 1 — NER JURÍDICO BRASILEIRO
# Dataset : peluz/lener_br (carregado direto do GitHub via CoNLL)
#           datasets>=4.5 não suporta mais loading scripts (.py)
#           → solução: baixar os .conll do repositório oficial
# Modelo  : neuralmind/bert-base-portuguese-cased
# Output  : PESSOA, ORGANIZAÇÃO, LOCAL, TEMPO, LEGISLAÇÃO, JURISPRUDÊNCIA
# Salva   : Google Drive → juris_models/ner_juridico
# ============================================================

# %%
# Verificar GPU
import torch
print("GPU disponível:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("VRAM:", round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1), "GB")
else:
    print("ATENÇÃO: sem GPU — vá em Runtime > Change runtime type > GPU T4")

# %%
# Instalar dependências
# !pip install transformers datasets seqeval accelerate -q

# %%
# Imports
import os
import urllib.request
import tempfile

import numpy as np
from datasets import Dataset, DatasetDict, Features, Sequence, ClassLabel, Value
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
)
from seqeval.metrics import f1_score, classification_report

# %%
# ── PASSO 1: Baixar os arquivos CoNLL diretamente do GitHub ──────────────────
# O HuggingFace Hub parou de executar scripts de loading (.py) no datasets>=4.5
# A solução é ler os arquivos raw do repositório oficial do LeNER-Br.

CONLL_URLS = {
    "train":      "https://raw.githubusercontent.com/peluz/lener-br/master/leNER-Br/train/train.conll",
    "validation": "https://raw.githubusercontent.com/peluz/lener-br/master/leNER-Br/dev/dev.conll",
    "test":       "https://raw.githubusercontent.com/peluz/lener-br/master/leNER-Br/test/test.conll",
}

def download_conll(url: str) -> str:
    """Baixa o arquivo e retorna o caminho local."""
    tmp = tempfile.NamedTemporaryFile(suffix=".conll", delete=False, mode="wb")
    urllib.request.urlretrieve(url, tmp.name)
    return tmp.name

def parse_conll(filepath: str) -> tuple[list[list[str]], list[list[str]]]:
    """
    Lê arquivo no formato CoNLL (token tag por linha, linhas vazias = fronteira).
    Retorna (lista_de_tokens, lista_de_tags).
    """
    all_tokens, all_tags = [], []
    tokens, tags = [], []

    with open(filepath, encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\n")
            if not line:
                if tokens:
                    all_tokens.append(tokens)
                    all_tags.append(tags)
                    tokens, tags = [], []
            else:
                parts = line.split()
                if len(parts) >= 2:
                    tokens.append(parts[0])
                    tags.append(parts[-1])  # última coluna é sempre a tag BIO

    if tokens:  # última sentença sem linha vazia final
        all_tokens.append(tokens)
        all_tags.append(tags)

    return all_tokens, all_tags

# Download e parse
raw_data = {}
for split, url in CONLL_URLS.items():
    path = download_conll(url)
    tkns, tgs = parse_conll(path)
    raw_data[split] = {"tokens": tkns, "tags": tgs}
    os.unlink(path)
    print(f"{split:12}: {len(tkns)} sentenças")

# %%
# ── PASSO 2: Construir o vocabulário de labels ────────────────────────────────
all_tags_flat = [t for split in raw_data.values() for row in split["tags"] for t in row]
label_names   = sorted(set(all_tags_flat))

# Garantir que "O" (outside) seja o índice 0 — convenção padrão
if "O" in label_names:
    label_names = ["O"] + [l for l in label_names if l != "O"]

id2label = {i: l for i, l in enumerate(label_names)}
label2id = {l: i for i, l in enumerate(label_names)}

print(f"\nLabels ({len(label_names)}):", label_names)

# %%
# ── PASSO 3: Converter tags string → int e montar DatasetDict ─────────────────
def build_hf_dataset(tokens_list, tags_list):
    return Dataset.from_dict({
        "tokens":   tokens_list,
        "ner_tags": [[label2id[t] for t in row] for row in tags_list],
    })

dataset = DatasetDict({
    split: build_hf_dataset(raw_data[split]["tokens"], raw_data[split]["tags"])
    for split in ["train", "validation", "test"]
})

print("\nDataset criado:")
print(dataset)
print("\nExemplo tokens :", dataset["train"][0]["tokens"][:10])
print("Exemplo ner_tags:", dataset["train"][0]["ner_tags"][:10])
print("Tags legíveis  :", [id2label[t] for t in dataset["train"][0]["ner_tags"][:10]])

# %%
# ── PASSO 4: Tokenização + alinhamento de labels ──────────────────────────────
MODEL_NAME = "neuralmind/bert-base-portuguese-cased"
tokenizer  = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_and_align_labels(examples):
    tokenized = tokenizer(
        examples["tokens"],
        truncation=True,
        max_length=512,
        is_split_into_words=True,
    )
    all_labels = []
    for i, labels in enumerate(examples["ner_tags"]):
        word_ids  = tokenized.word_ids(batch_index=i)
        aligned   = []
        prev_word = None
        for word_id in word_ids:
            if word_id is None:
                aligned.append(-100)          # tokens especiais [CLS], [SEP], [PAD]
            elif word_id != prev_word:
                aligned.append(labels[word_id])
            else:
                aligned.append(-100)          # subtokens: ignorar no loss
            prev_word = word_id
        all_labels.append(aligned)
    tokenized["labels"] = all_labels
    return tokenized

tokenized_ds = dataset.map(
    tokenize_and_align_labels,
    batched=True,
    remove_columns=dataset["train"].column_names,
)
print("Tokenização concluída.")

# %%
# ── PASSO 5: Modelo e data collator ───────────────────────────────────────────
model = AutoModelForTokenClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(label_names),
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True,
)

data_collator = DataCollatorForTokenClassification(
    tokenizer, pad_to_multiple_of=8
)

# %%
# ── PASSO 6: Métricas ─────────────────────────────────────────────────────────
def compute_metrics(p):
    preds, labels = p
    preds = np.argmax(preds, axis=-1)

    true_labels = [
        [id2label[l] for l in row if l != -100]
        for row in labels
    ]
    true_preds = [
        [id2label[pred] for pred, l in zip(pred_row, label_row) if l != -100]
        for pred_row, label_row in zip(preds, labels)
    ]
    return {"f1": f1_score(true_labels, true_preds)}

# %%
# ── PASSO 7: Treinar ─────────────────────────────────────────────────────────
training_args = TrainingArguments(
    output_dir="./ner_juridico_output",
    num_train_epochs=5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    learning_rate=2e-5,
    weight_decay=0.01,
    warmup_steps=50,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    fp16=torch.cuda.is_available(),
    report_to="none",
    logging_steps=20,
    dataloader_num_workers=2,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_ds["train"],
    eval_dataset=tokenized_ds["validation"],
    data_collator=data_collator,
    processing_class=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()

# %%
# ── PASSO 8: Avaliar no test set ──────────────────────────────────────────────
test_results = trainer.evaluate(tokenized_ds["test"])
print("=== Resultado no test set ===")
for k, v in test_results.items():
    print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

preds_output = trainer.predict(tokenized_ds["test"])
preds  = np.argmax(preds_output.predictions, axis=-1)
labels = preds_output.label_ids

true_labels = [[id2label[l] for l in row if l != -100] for row in labels]
true_preds  = [
    [id2label[p] for p, l in zip(pr, lr) if l != -100]
    for pr, lr in zip(preds, labels)
]
print("\n=== Relatório por entidade ===")
print(classification_report(true_labels, true_preds))

# %%
# ── PASSO 9: Salvar no Google Drive ──────────────────────────────────────────
import shutil, json
from google.colab import files

SAVE_DIR = "./hf_models/ner_juridico"
trainer.save_model(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)

with open(f"{SAVE_DIR}/label_map.json", "w") as f:
    json.dump({"id2label": id2label, "label2id": label2id}, f, ensure_ascii=False, indent=2)

shutil.make_archive(SAVE_DIR, "zip", SAVE_DIR)
files.download("ner_juridico.zip")
print("Download iniciado: ner_juridico.zip")

# %%
# ── PASSO 10: Teste de inferência ─────────────────────────────────────────────
from transformers import pipeline

ner_pipe = pipeline(
    "ner",
    model=SAVE_DIR,
    tokenizer=SAVE_DIR,
    aggregation_strategy="simple",
    device=0 if torch.cuda.is_available() else -1,
)

texto_teste = (
    "O contrato celebrado entre PETROBRAS S.A. e TECH SOLUTIONS LTDA, "
    "em 15 de março de 2024, com sede em São Paulo, prevê penalidade de 30% "
    "conforme o art. 12 da Lei nº 8.666/93, conforme decidido no REsp 1.234.567/SP."
)

print("Texto:", texto_teste)
print("\nEntidades detectadas:")
for ent in ner_pipe(texto_teste):
    print(f"  [{ent['entity_group']:20}] {ent['word']:45} score={ent['score']:.3f}")
