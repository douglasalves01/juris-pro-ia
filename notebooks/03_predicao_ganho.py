
!pip install transformers datasets accelerate scikit-learn requests -q

# %% [markdown]
# ## 2. Imports

# %%
import torch
import numpy as np
import pandas as pd
import requests
from datasets import load_dataset, DatasetDict, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from sklearn.metrics import classification_report, f1_score as sk_f1, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

# %%
# ── HELPER: carrega dataset com fallback para parquet do Hub ─────────────────
# datasets>=4.0 não suporta mais loading scripts (.py) — este helper
# tenta o load normal e, se falhar, busca os parquet exportados pelo Hub.

def load_dataset_robust(name: str, **kwargs) -> DatasetDict:
    try:
        return load_dataset(name, **kwargs)
    except RuntimeError as e:
        if "no longer supported" not in str(e):
            raise
        print(f"[fallback] loading script bloqueado — carregando parquet do Hub...")
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
# ## 3. Carregar e explorar dataset

# %%
raw = load_dataset_robust("joelniklaus/brazilian_court_decisions")
print(raw)
print("\nColunas:", raw["train"].column_names)
print("\nExemplo:")
print(raw["train"][0])

# Distribuição de labels
from collections import Counter
for split in ["train", "validation", "test"]:
    c = Counter(raw[split]["judgment_label"])
    print(f"\n{split}: {dict(c)}")

# %% [markdown]
# ## 4. Mapear labels → 3 classes significativas

# %%
# Mapeamento original → significado prático para o advogado
# "reforma"        → cliente que recorreu GANHOU
# "manteve"        → cliente que recorreu PERDEU
# "não_conhecido"  → recurso não admitido (procedural / inconclusivo)

LABEL_MAP = {
    "reforma":        0,  # ganhou
    "manteve":        1,  # perdeu
    "não_conhecido":  2,  # inconclusivo
}
ID2LABEL = {0: "ganhou", 1: "perdeu", 2: "inconclusivo"}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}

def map_labels(example):
    example["label"] = LABEL_MAP.get(example["judgment_label"], 2)
    return example

ds = raw.map(map_labels)
print("\nLabels mapeadas:", Counter(ds["train"]["label"]))

# %% [markdown]
# ## 5. Tokenização

# %%
MODEL_NAME = "neuralmind/bert-base-portuguese-cased"
tokenizer  = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize(examples):
    return tokenizer(
        examples["decision_description"],
        truncation=True,
        max_length=512,
        padding=False,
    )

tok_ds = ds.map(tokenize, batched=True, remove_columns=ds["train"].column_names)
# Recolocar labels
for split in ["train", "validation", "test"]:
    tok_ds[split] = tok_ds[split].add_column("label", ds[split]["label"])

data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)

# %% [markdown]
# ## 6. Peso de classes (dataset pode ser desbalanceado)

# %%

train_labels = ds["train"]["label"]
classes_presentes = np.unique(train_labels)
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=classes_presentes,
    y=train_labels
)
# Garante alinhamento dos pesos com as classes do modelo (0,1,2)
class_weights_full = np.ones(3, dtype=np.float32)
for idx, c in enumerate(classes_presentes):
    class_weights_full[c] = class_weights[idx]
class_weights_tensor = torch.tensor(class_weights_full, dtype=torch.float)
print("Pesos por classe:", class_weights_full)

# %% [markdown]
# ## 7. Modelo com weighted loss

# %%
from torch.nn import CrossEntropyLoss

class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits  = outputs.logits
        loss_fn = CrossEntropyLoss(
            weight=class_weights_tensor.to(logits.device)
        )
        loss = loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=3,
    id2label=ID2LABEL,
    label2id=LABEL2ID,
    ignore_mismatched_sizes=True,
)

# %% [markdown]
# ## 8. Métricas

# %%
def compute_metrics(p):
    preds  = np.argmax(p.predictions, axis=-1)
    labels = p.label_ids
    return {
        "accuracy":    float((preds == labels).mean()),
        "f1_macro":    sk_f1(labels, preds, average="macro",    zero_division=0),
        "f1_weighted": sk_f1(labels, preds, average="weighted", zero_division=0),
    }

# %% [markdown]
# ## 9. Treinar

# %%
training_args = TrainingArguments(
    output_dir="./predicao_ganho_output",
    num_train_epochs=4,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    learning_rate=2e-5,
    weight_decay=0.01,
    warmup_steps=100,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",
    greater_is_better=True,
    fp16=torch.cuda.is_available(),
    report_to="none",
    logging_steps=50,
)

trainer = WeightedTrainer(
    model=model,
    args=training_args,
    train_dataset=tok_ds["train"],
    eval_dataset=tok_ds["validation"],
    data_collator=data_collator,
    processing_class=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()

# %% [markdown]
# ## 10. Avaliar

# %%
results = trainer.evaluate(tok_ds["test"])
print("=== Resultado no test set ===")
for k, v in results.items():
    print(f"  {k}: {round(v,4) if isinstance(v,float) else v}")

preds_out  = trainer.predict(tok_ds["test"])
preds      = np.argmax(preds_out.predictions, axis=-1)
true_labels = preds_out.label_ids

print("\n=== Relatório por classe ===")
print(classification_report(
    true_labels, preds,
    labels=[0, 1, 2],
    target_names=["ganhou", "perdeu", "inconclusivo"],
    zero_division=0
))

print("=== Matriz de confusão ===")
print(confusion_matrix(true_labels, preds))

# %% [markdown]
# ## 11. Salvar no Google Drive

# %%
import shutil
from google.colab import files

SAVE_DIR = "./hf_models/predicao_ganho"
trainer.save_model(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)

shutil.make_archive(SAVE_DIR, "zip", SAVE_DIR)
files.download("predicao_ganho.zip")
print("Download iniciado: predicao_ganho.zip")

# %% [markdown]
# ## 12. Inferência com probabilidades

# %%
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer_inf = AutoTokenizer.from_pretrained(SAVE_DIR)
model_inf     = AutoModelForSequenceClassification.from_pretrained(SAVE_DIR)
model_inf.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_inf.to(device)

def prever_chance_ganho(texto: str) -> dict:
    """
    Retorna probabilidade de ganho, perda e resultado inconclusivo.
    Usado pelo endpoint /analysis/{id} para preencher win_probability.
    """
    enc = tokenizer_inf(
        texto, return_tensors="pt",
        truncation=True, max_length=512
    ).to(device)

    with torch.no_grad():
        logits = model_inf(**enc).logits
        probs  = F.softmax(logits, dim=-1)[0].cpu().tolist()

    resultado = {
        "ganhou":       round(probs[0], 4),
        "perdeu":       round(probs[1], 4),
        "inconclusivo": round(probs[2], 4),
        "previsao":     ID2LABEL[int(np.argmax(probs))],
        "confianca":    round(max(probs), 4),
    }
    return resultado

# Teste
casos = [
    "A parte autora comprovou cabalmente o dano sofrido e o nexo causal com "
     "a conduta da ré, sendo devida a indenização por danos morais.",
    "O recurso não demonstrou qualquer violação às normas do Código de Defesa "
     "do Consumidor, mantendo-se a sentença recorrida.",
    "O pedido foi julgado improcedente por insuficiência probatória, cabendo "
     "ao autor arcar com as custas processuais.",
]

print("=== Predições de Chance de Ganho ===")
for c in casos:
    r = prever_chance_ganho(c)
    print(f"\n  Texto: {c[:80]}...")
    print(f"  Previsão   : {r['previsao'].upper()} (confiança: {r['confianca']:.1%})")
    print(f"  Ganhou     : {r['ganhou']:.1%}")
    print(f"  Perdeu     : {r['perdeu']:.1%}")
    print(f"  Inconclusivo: {r['inconclusivo']:.1%}")
