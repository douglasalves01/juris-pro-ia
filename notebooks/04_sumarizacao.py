
!pip install transformers datasets accelerate sentencepiece rouge-score requests -q

# %% [markdown]
# ## 2. Imports

# %%
import torch
import numpy as np
import requests
from datasets import load_dataset, DatasetDict, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    GenerationConfig,
)
import nltk
nltk.download("punkt", quiet=True)
try:
    from rouge_score import rouge_scorer
except ModuleNotFoundError:
    import sys
    import subprocess

    subprocess.check_call([sys.executable, "-m", "pip", "install", "rouge-score", "-q"])
    from rouge_score import rouge_scorer

# %%
# ── HELPER: fallback para parquet do Hub ────────────────────────────────────

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
# ## 3. Carregar datasets
#
# Usamos `decision_description` (ementa) como **alvo** (summary).
# Como o dataset não tem o texto completo separado, simulamos o input
# concatenando a ementa com contexto adicional — em produção, o PDF
# completo do processo entra aqui.

# %%
raw = load_dataset_robust("joelniklaus/brazilian_court_decisions")
print(raw)
print("\nExemplo ementa:", raw["train"][0]["decision_description"][:300])

# %% [markdown]
# ## 4. Construir pares (input → target)
#
# Estratégia realista:
#   - input  = versão expandida (simulada): ementa + ruído/contexto extra
#   - target = ementa original (resumo gold)
#
# Em produção, input = texto completo do PDF extraído.

# %%
import random
random.seed(42)

PREFIXO = "resuma o processo jurídico: "

# Frases de contexto jurídico para simular a "extensão" do documento real
CONTEXTO_EXTRA = [
    "Trata-se de ação indenizatória ajuizada perante a 3ª Vara Cível. "
    "O autor alega ter sofrido danos materiais e morais em razão da conduta da ré. "
    "A parte contrária apresentou contestação refutando todos os pedidos. "
    "Realizou-se audiência de instrução com oitiva das partes e testemunhas. ",

    "O feito teve regular tramitação com observância do contraditório e ampla defesa. "
    "Foram juntados documentos comprobatórios e laudos periciais. "
    "O Ministério Público emitiu parecer. O juiz prolatou sentença. ",

    "Cuida-se de recurso de apelação interposto pela parte recorrente "
    "insurgindo-se contra a r. sentença de primeiro grau. "
    "As razões recursais foram devidamente apresentadas e contrarrazões ofertadas. ",

    "A matéria foi submetida ao colegiado para apreciação. "
    "O relator proferiu seu voto após análise detalhada dos autos. "
    "Os demais desembargadores acompanharam o relator. ",
]

def build_pairs(examples):
    inputs  = []
    targets = []
    for ementa in examples["decision_description"]:
        if not ementa or len(ementa.strip()) < 50:
            continue
        ctx = random.choice(CONTEXTO_EXTRA)
        # Input: contexto + ementa embaralhada (simula doc longo)
        palavras = ementa.split()
        random.shuffle(palavras)
        texto_longo = ctx + " ".join(palavras)
        inputs.append(PREFIXO + texto_longo[:1500])
        targets.append(ementa.strip())
    return {"input_text": inputs, "target_text": targets}

# Aplicar
train_pairs = build_pairs({"decision_description": raw["train"]["decision_description"]})
val_pairs   = build_pairs({"decision_description": raw["validation"]["decision_description"]})
test_pairs  = build_pairs({"decision_description": raw["test"]["decision_description"]})

train_ds_raw = Dataset.from_dict(train_pairs)
val_ds_raw   = Dataset.from_dict(val_pairs)
test_ds_raw  = Dataset.from_dict(test_pairs)

print(f"Treino: {len(train_ds_raw)} | Val: {len(val_ds_raw)} | Teste: {len(test_ds_raw)}")
print("\nExemplo input :", train_ds_raw[0]["input_text"][:200])
print("Exemplo target:", train_ds_raw[0]["target_text"][:200])

# %% [markdown]
# ## 5. Tokenização

# %%
MODEL_NAME = "unicamp-dl/ptt5-base-portuguese-vocab"

tokenizer  = AutoTokenizer.from_pretrained(MODEL_NAME)

MAX_INPUT  = 512
MAX_TARGET = 256

def preprocess(examples):
    model_inputs = tokenizer(
        examples["input_text"],
        max_length=MAX_INPUT,
        truncation=True,
        padding=False,
    )
    labels = tokenizer(
        examples["target_text"],
        max_length=MAX_TARGET,
        truncation=True,
        padding=False,
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tok_train = train_ds_raw.map(preprocess, batched=True,
                              remove_columns=["input_text", "target_text"])
tok_val   = val_ds_raw.map(preprocess,   batched=True,
                            remove_columns=["input_text", "target_text"])
tok_test  = test_ds_raw.map(preprocess,  batched=True,
                             remove_columns=["input_text", "target_text"])

print("Tokenização concluída.")

# %% [markdown]
# ## 6. Modelo e data collator

# %%
model         = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
# Definir decoder_start_token_id e bos_token_id corretamente para o modelo
if getattr(model.config, 'decoder_start_token_id', None) is None:
    model.config.decoder_start_token_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.pad_token_id
if getattr(model.config, 'bos_token_id', None) is None:
    model.config.bos_token_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.pad_token_id
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, pad_to_multiple_of=8)

# %% [markdown]
# ## 7. Métricas ROUGE

# %%
scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False)

def compute_metrics(eval_preds):
    preds, labels = eval_preds

    # Decodificar predições
    if isinstance(preds, tuple):
        preds = preds[0]
    preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
    decoded_preds  = tokenizer.batch_decode(preds,  skip_special_tokens=True)

    # Decodificar labels
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    r1s, r2s, rLs = [], [], []
    for pred, label in zip(decoded_preds, decoded_labels):
        scores = scorer.score(label, pred)
        r1s.append(scores["rouge1"].fmeasure)
        r2s.append(scores["rouge2"].fmeasure)
        rLs.append(scores["rougeL"].fmeasure)

    return {
        "rouge1": round(np.mean(r1s), 4),
        "rouge2": round(np.mean(r2s), 4),
        "rougeL": round(np.mean(rLs), 4),
    }

# %% [markdown]
# ## 8. Treinar

# %%
gen_config = GenerationConfig(
    max_new_tokens=MAX_TARGET,
    num_beams=4,
    early_stopping=True,
    no_repeat_ngram_size=3,
)

training_args = Seq2SeqTrainingArguments(
    output_dir="./sumarizacao_output",
    num_train_epochs=5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=5e-5,
    weight_decay=0.01,
    warmup_steps=100,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="rougeL",
    greater_is_better=True,
    predict_with_generate=True,
    generation_config=gen_config,
    fp16=torch.cuda.is_available(),
    report_to="none",
    logging_steps=50,
    dataloader_num_workers=2,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tok_train,
    eval_dataset=tok_val,
    data_collator=data_collator,
    processing_class=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()

# %% [markdown]
# ## 9. Avaliar

# %%
results = trainer.evaluate(tok_test, metric_key_prefix="test")
print("=== Resultado no teste ===")
for k, v in results.items():
    print(f"  {k}: {round(v,4) if isinstance(v,float) else v}")

# %% [markdown]
# ## 10. Salvar

# %%
import shutil
from google.colab import files

SAVE_DIR = "./hf_models/sumarizacao"
trainer.save_model(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)
gen_config.save_pretrained(SAVE_DIR)

shutil.make_archive(SAVE_DIR, "zip", SAVE_DIR)
files.download("sumarizacao.zip")
print("Download iniciado: sumarizacao.zip")

# %% [markdown]
# ## 11. Inferência — geração de resumo executivo

# %%
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

sum_pipe = pipeline(
    "summarization",
    model=SAVE_DIR,
    tokenizer=SAVE_DIR,
    device=0 if torch.cuda.is_available() else -1,
)

textos_teste = [
    (
        "resuma o processo jurídico: "
        "Trata-se de ação indenizatória movida por consumidor em face de operadora "
        "de telefonia. O autor alega que teve seu nome negativado indevidamente após "
        "o cancelamento do contrato sem sua anuência. A ré não comprovou a regularidade "
        "da cobrança. O juiz sentenciou procedente o pedido, condenando a ré ao pagamento "
        "de R$ 8.000,00 a título de danos morais e determinando a exclusão da negativação."
    ),
    (
        "resuma o processo jurídico: "
        "Recurso de apelação em ação trabalhista. O reclamante pleiteia reconhecimento "
        "de horas extras e adicional de insalubridade. A empresa alegou que o horário "
        "era controlado por ponto eletrônico. Perícia demonstrou irregularidades no "
        "sistema de registro. O Tribunal reformou parcialmente a sentença, reconhecendo "
        "as horas extras mas indeferindo o adicional de insalubridade por ausência de prova."
    ),
]

print("=== Resumos gerados ===")
for texto in textos_teste:
    resultado = sum_pipe(
        texto,
        max_new_tokens=200,
        min_length=40,
        num_beams=4,
        no_repeat_ngram_size=3,
        truncation=True,
    )
    print(f"\nInput : {texto[40:150]}...")
    print(f"Resumo: {resultado[0]['summary_text']}")
    print("-" * 80)
