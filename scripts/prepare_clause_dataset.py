"""Prepara dataset JSONL para fine-tune de classificacao de clausulas.

Entrada esperada: CSV com colunas `texto` e `label`.
Labels aceitos: abusiva, padrao, favoravel.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


_LABELS = {"abusiva", "padrao", "favoravel"}


def prepare(input_csv: Path, output_jsonl: Path) -> int:
    rows = 0
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with input_csv.open(newline="", encoding="utf-8") as source, output_jsonl.open("w", encoding="utf-8") as target:
        reader = csv.DictReader(source)
        if "texto" not in (reader.fieldnames or []) or "label" not in (reader.fieldnames or []):
            raise ValueError("CSV precisa conter colunas 'texto' e 'label'.")
        for row in reader:
            text = (row.get("texto") or "").strip()
            label = (row.get("label") or "").strip().lower()
            if not text or label not in _LABELS:
                continue
            target.write(json.dumps({"text": text, "label": label}, ensure_ascii=False) + "\n")
            rows += 1
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_csv", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("datasets/clausulas/clausulas.jsonl"),
    )
    args = parser.parse_args()
    count = prepare(args.input_csv, args.output)
    print(f"{count} exemplos gravados em {args.output}")


if __name__ == "__main__":
    main()
