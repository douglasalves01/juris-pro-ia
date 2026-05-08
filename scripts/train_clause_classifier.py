"""Fine-tune de classificador de clausulas abusiva/padrao/favoravel."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)


LABEL_MAP = {"abusiva": 0, "padrao": 1, "favoravel": 2}


def train(
    train_jsonl: Path,
    output_dir: Path,
    model_name: str = "microsoft/deberta-v3-small",
    epochs: float = 3.0,
) -> None:
    dataset = load_dataset("json", data_files=str(train_jsonl))["train"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def encode(batch):
        encoded = tokenizer(batch["text"], truncation=True, max_length=384)
        encoded["labels"] = [LABEL_MAP[label] for label in batch["label"]]
        return encoded

    tokenized = dataset.map(encode, batched=True, remove_columns=dataset.column_names)
    split = tokenized.train_test_split(test_size=0.15, seed=42)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(LABEL_MAP),
        id2label={idx: label for label, idx in LABEL_MAP.items()},
        label2id=LABEL_MAP,
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return {"accuracy": float((predictions == labels).mean())}

    args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        learning_rate=2e-5,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        report_to=[],
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=split["train"],
        eval_dataset=split["test"],
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )
    trainer.train()
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    (output_dir / "label_map.json").write_text(json.dumps(LABEL_MAP, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("train_jsonl", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("hf_models/classificacao_clausulas"))
    parser.add_argument("--model-name", default="microsoft/deberta-v3-small")
    parser.add_argument("--epochs", type=float, default=3.0)
    args = parser.parse_args()
    train(args.train_jsonl, args.output_dir, args.model_name, args.epochs)


if __name__ == "__main__":
    main()
