from __future__ import annotations

import csv

from api.ml.models.clause_classifier import classify_clause, classify_clauses
from scripts.prepare_clause_dataset import prepare


def test_clause_classifier_detecta_multa_abusiva() -> None:
    result = classify_clause(
        {
            "numero": "1",
            "tipo": "multa",
            "titulo": "Multa",
            "texto": "Em caso de atraso, aplica-se multa de 35% sobre o valor total.",
        }
    )

    assert result.label == "abusiva"
    assert result.confidence >= 0.8
    assert "35%" in result.rationale


def test_clause_classifier_detecta_clausula_favoravel() -> None:
    result = classify_clause(
        {
            "numero": "2",
            "tipo": "rescisão",
            "titulo": "Rescisao",
            "texto": "As partes observarao boa-fe e prazo de aviso de 60 dias para rescisao.",
        }
    )

    assert result.label == "favoravel"


def test_classify_clauses_preserva_quantidade() -> None:
    clauses = [
        {"numero": "1", "texto": "Clausula padrao de pagamento em ate 10 dias."},
        {"numero": "2", "texto": "Multa de 25% em caso de atraso."},
    ]

    assert len(classify_clauses(clauses)) == len(clauses)


def test_prepare_clause_dataset_filtra_labels_invalidos(tmp_path) -> None:
    source = tmp_path / "clauses.csv"
    output = tmp_path / "clauses.jsonl"
    with source.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["texto", "label"])
        writer.writeheader()
        writer.writerow({"texto": "Multa de 35%.", "label": "abusiva"})
        writer.writerow({"texto": "Texto sem label valido.", "label": "ignorar"})

    count = prepare(source, output)

    assert count == 1
    assert '"label": "abusiva"' in output.read_text(encoding="utf-8")
