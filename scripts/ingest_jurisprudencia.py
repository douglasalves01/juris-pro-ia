"""Ingestao de jurisprudencia para monitoramento.

Aceita JSON/JSONL local ou uma URL que retorne lista JSON. O script normaliza as
decisoes e grava JSONL pronto para importacao por jobs de embedding/Qdrant.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import httpx

from api.services.jurisprudence_monitor import normalize_decision


def _load_source(source: str) -> list[dict[str, Any]]:
    if source.startswith("http://") or source.startswith("https://"):
        response = httpx.get(source, timeout=30.0)
        response.raise_for_status()
        data = response.json()
        if isinstance(data, dict):
            data = data.get("items") or data.get("decisions") or []
        return [item for item in data if isinstance(item, dict)]

    path = Path(source)
    if path.suffix.lower() == ".jsonl":
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("items") or data.get("decisions") or []
    return [item for item in data if isinstance(item, dict)]


def ingest(source: str, output: Path) -> int:
    decisions = [normalize_decision(item) for item in _load_source(source)]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as file:
        for decision in decisions:
            file.write(json.dumps(decision, ensure_ascii=False) + "\n")
    return len(decisions)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", help="Arquivo JSON/JSONL ou URL de jurisprudencia")
    parser.add_argument("--output", type=Path, default=Path("data/jurisprudencia/decisions.jsonl"))
    args = parser.parse_args()
    count = ingest(args.source, args.output)
    print(f"{count} decisoes normalizadas em {args.output}")


if __name__ == "__main__":
    main()
