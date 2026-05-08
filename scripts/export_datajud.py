"""
Script pronto para copiar e colar no Google Colab.

No Colab:
1. Crie um arquivo/celula com este script inteiro.
2. Troque DATAJUD_API_KEY = "COLE_SUA_CHAVE_AQUI".
3. Aperte Run.

O script instala dependencias, busca processos publicos do DataJud,
gera um texto de similaridade com classe/assuntos/movimentos,
cria um ZIP e tenta baixar automaticamente no Colab.

Varios tribunais: padrao sao 32 downloads em paralelo (--workers); cada
tribunal ainda pagina a API em serie (search_after). Ajuste --workers se
a API limitar (429 etc.).
"""
!pip install -q requests torch sentence-transformers transformers
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time
import uuid
import zipfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DATAJUD_API_KEY = "cDZHYzlZa0JadVREZDJCendQbXY6SkJlTzNjLV9TRENyQk1RdnFKZGRQdw=="
TRIBUNAIS = ["TJSP"]
PRESET = "comercial"
MAX_POR_TRIBUNAL = 500
PAGE_SIZE = 100
SLEEP_ENTRE_PAGINAS = 0.3
OUTPUT_DIR = "datajud_export"
ZIP_NAME = None
GERAR_EMBEDDINGS = True
BAIXAR_ZIP_NO_COLAB = True
DEFAULT_HF_MODEL = "rufimelo/Legal-BERTimbau-sts-large"
# Tribunais em paralelo (I/O). Suba com --workers se a API nao limitar; baixe em caso de 429/erros.
DEFAULT_FETCH_WORKERS = 32


def ensure_dependencies() -> None:
    missing = []
    modules = {
        "requests": "requests",
        "torch": "torch",
        "sentence_transformers": "sentence-transformers",
        "transformers": "transformers",
    }
    for module, package in modules.items():
        if importlib.util.find_spec(module) is None:
            missing.append(package)
    if missing:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", *missing]
        )


ensure_dependencies()

import requests
import torch
from sentence_transformers import SentenceTransformer

DATAJUD_BASE = "https://api-publica.datajud.cnj.jus.br"
BATCH_SIZE = 64

TRIBUNAIS_COMERCIAIS = [
    "TJSP",
    "TJRJ",
    "TJMG",
    "TJRS",
    "TJPR",
    "TJSC",
    "TJBA",
    "TJPE",
    "TJGO",
    "TJDFT",
    "STJ",
    "TRF1",
    "TRF3",
    "TRF4",
    "TRT2",
    "TRT3",
    "TRT4",
    "TRT15",
]

TRIBUNAIS_TODOS = [
    "TST",
    "TSE",
    "STJ",
    "STM",
    "TRF1",
    "TRF2",
    "TRF3",
    "TRF4",
    "TRF5",
    "TRF6",
    "TJAC",
    "TJAL",
    "TJAM",
    "TJAP",
    "TJBA",
    "TJCE",
    "TJDFT",
    "TJES",
    "TJGO",
    "TJMA",
    "TJMG",
    "TJMS",
    "TJMT",
    "TJPA",
    "TJPB",
    "TJPE",
    "TJPI",
    "TJPR",
    "TJRJ",
    "TJRN",
    "TJRO",
    "TJRR",
    "TJRS",
    "TJSC",
    "TJSE",
    "TJSP",
    "TJTO",
    "TRT1",
    "TRT2",
    "TRT3",
    "TRT4",
    "TRT5",
    "TRT6",
    "TRT7",
    "TRT8",
    "TRT9",
    "TRT10",
    "TRT11",
    "TRT12",
    "TRT13",
    "TRT14",
    "TRT15",
    "TRT16",
    "TRT17",
    "TRT18",
    "TRT19",
    "TRT20",
    "TRT21",
    "TRT22",
    "TRT23",
    "TRT24",
]

OUTCOME_MAP = {
    "procedente": "procedente",
    "improcedente": "improcedente",
    "parcialmente procedente": "parcialmente procedente",
    "provido": "procedente",
    "não provido": "improcedente",
    "nao provido": "improcedente",
    "desprovido": "improcedente",
    "negado provimento": "improcedente",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tribunal", action="append", default=None, help="Ex: TJSP. Pode repetir.")
    parser.add_argument("--preset", choices=("comercial", "todos"), default=PRESET)
    parser.add_argument("--max", type=int, default=MAX_POR_TRIBUNAL, help="Máximo por tribunal.")
    parser.add_argument("--size", type=int, default=PAGE_SIZE, help="Tamanho da página da API.")
    parser.add_argument("--sleep", type=float, default=SLEEP_ENTRE_PAGINAS, help="Pausa entre páginas.")
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--zip-name", default=ZIP_NAME)
    parser.add_argument("--no-embeddings", action="store_true", default=not GERAR_EMBEDDINGS)
    parser.add_argument("--model-path", default=DEFAULT_HF_MODEL)
    parser.add_argument("--no-download", action="store_true", default=not BAIXAR_ZIP_NO_COLAB)
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_FETCH_WORKERS,
        help=f"Tribunais em paralelo (padrao {DEFAULT_FETCH_WORKERS}). Reduza se aparecer limite da API.",
    )
    args, _unknown = parser.parse_known_args()

    api_key = (DATAJUD_API_KEY if DATAJUD_API_KEY != "COLE_SUA_CHAVE_AQUI" else "").strip()
    api_key = os.getenv("DATAJUD_API_KEY", api_key).strip()
    if not api_key:
        raise SystemExit(
            'ERRO: edite o script e troque DATAJUD_API_KEY = "COLE_SUA_CHAVE_AQUI".'
        )

    tribunais = args.tribunal if args.tribunal is not None else TRIBUNAIS
    if not tribunais:
        tribunais = TRIBUNAIS_TODOS if args.preset == "todos" else TRIBUNAIS_COMERCIAIS
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    tribunais_norm = [t.upper().strip() for t in tribunais]
    workers = max(1, args.workers)

    if workers == 1 or len(tribunais_norm) == 1:
        for tribunal in tribunais_norm:
            print(f"\n-- {tribunal} --")
            records.extend(
                fetch_acordaos(
                    tribunal=tribunal,
                    api_key=api_key,
                    max_results=args.max,
                    size=args.size,
                    sleep_seconds=args.sleep,
                )
            )
    else:
        print(f"\nBaixando {len(tribunais_norm)} tribunal(is) com até {workers} threads...")

        def _one(t: str) -> list[dict[str, Any]]:
            return fetch_acordaos(
                tribunal=t,
                api_key=api_key,
                max_results=args.max,
                size=args.size,
                sleep_seconds=args.sleep,
                log_tag=t,
            )

        with ThreadPoolExecutor(max_workers=min(workers, len(tribunais_norm))) as executor:
            future_to_tribunal = {executor.submit(_one, t): t for t in tribunais_norm}
            for future in as_completed(future_to_tribunal):
                t = future_to_tribunal[future]
                try:
                    part = future.result()
                    records.extend(part)
                    print(f"  [ok] {t}: {len(part)} registros")
                except Exception as exc:
                    print(f"  [erro] {t}: {exc}")

    records = dedupe_records(records)
    print(f"\nRegistros unicos: {len(records)}")

    if not args.no_embeddings:
        add_embeddings(records, args.model_path)

    manifest = build_manifest(records, tribunais, args)
    data_path = out_dir / "casos_datajud.json"
    manifest_path = out_dir / "manifest.json"
    with data_path.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False)
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    zip_name = args.zip_name or f"datajud_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
    zip_path = out_dir / zip_name
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(data_path, data_path.name)
        zf.write(manifest_path, manifest_path.name)

    print(f"\nJSON: {data_path}")
    print(f"Manifest: {manifest_path}")
    print(f"ZIP: {zip_path}")
    maybe_download_colab(zip_path, args.no_download)


def fetch_acordaos(
    tribunal: str,
    api_key: str,
    max_results: int,
    size: int,
    sleep_seconds: float,
    log_tag: str | None = None,
) -> list[dict[str, Any]]:
    # Uma Session por chamada (roda numa thread): reuse de TCP/TLS entre páginas.
    session = requests.Session()
    prefix = f"[{log_tag}] " if log_tag else ""
    headers = {"Authorization": f"APIKey {api_key}"}
    endpoint = f"{DATAJUD_BASE}/api_publica_{tribunal.lower()}/_search"
    body: dict[str, Any] = {
        "size": size,
        "query": {"bool": {"must": [{"exists": {"field": "numeroProcesso"}}]}},
        "_source": [
            "id",
            "tribunal",
            "numeroProcesso",
            "dataAjuizamento",
            "dataHoraUltimaAtualizacao",
            "@timestamp",
            "grau",
            "nivelSigilo",
            "formato",
            "sistema",
            "classe",
            "assuntos",
            "orgaoJulgador",
            "movimentos",
            "ementa",
            "tipoDocumento",
        ],
        "sort": [{"@timestamp": {"order": "desc"}}, {"dataAjuizamento": {"order": "desc"}}],
    }
    records: list[dict[str, Any]] = []
    fetched = 0
    search_after = None

    while fetched < max_results:
        if search_after:
            body["search_after"] = search_after
        try:
            resp = session.post(endpoint, json=body, headers=headers, timeout=30)
            resp.raise_for_status()
        except requests.HTTPError as exc:
            print(f"{prefix}HTTP {exc.response.status_code}: {exc}")
            break
        except Exception as exc:
            print(f"{prefix}Erro: {exc}")
            break

        hits = resp.json().get("hits", {}).get("hits", [])
        if not hits:
            break

        for hit in hits:
            record = normalize_hit(hit, tribunal)
            if record:
                records.append(record)

        fetched += len(hits)
        search_after = hits[-1].get("sort")
        if log_tag:
            if fetched % 100 == 0 or fetched >= max_results:
                print(f"{prefix}coletados: {fetched} | validos: {len(records)}", flush=True)
        else:
            print(f"  coletados: {fetched} | validos: {len(records)}", end="\r")
        time.sleep(sleep_seconds)

    print(f"{prefix}coletados: {fetched} | validos: {len(records)}")
    return records


def normalize_hit(hit: dict[str, Any], tribunal: str) -> dict[str, Any] | None:
    src = hit.get("_source") or {}
    movimentos = normalize_movimentos(src.get("movimentos") or [])
    assuntos = normalize_named_list(src.get("assuntos") or [])
    classe = src.get("classe") or {}
    orgao = src.get("orgaoJulgador") or {}
    numero = str(src.get("numeroProcesso") or "")
    resumo = build_similarity_text(src, assuntos, movimentos)
    if len(resumo) < 80:
        return None
    data_movimento = movimentos[0].get("dataHora") if movimentos else None

    return {
        "id": str(uuid.uuid5(uuid.NAMESPACE_URL, f"{tribunal}:{numero}")),
        "data_source": "datajud",
        "datajud_id": src.get("id") or hit.get("_id"),
        "tribunal": str(src.get("tribunal") or tribunal),
        "numeroProcesso": numero,
        "dataAjuizamento": src.get("dataAjuizamento"),
        "dataUltimoMovimento": data_movimento,
        "dataHoraUltimaAtualizacao": src.get("dataHoraUltimaAtualizacao"),
        "grau": src.get("grau"),
        "nivelSigilo": src.get("nivelSigilo"),
        "formato": src.get("formato"),
        "sistema": src.get("sistema"),
        "classe_codigo": classe.get("codigo"),
        "classe_nome": classe.get("nome"),
        "assuntos": assuntos,
        "orgaoJulgador_codigo": orgao.get("codigo"),
        "orgaoJulgador_nome": orgao.get("nome"),
        "orgaoJulgador_municipio_ibge": orgao.get("codigoMunicipioIBGE"),
        "tipo": infer_tipo(resumo, assuntos, str(classe.get("nome") or "")),
        "outcome": infer_outcome(movimentos),
        "titulo": build_title(src, assuntos),
        "resumo": resumo,
        "movimentos_relevantes": movimentos[:20],
    }


def normalize_named_list(items: list[Any]) -> list[dict[str, Any]]:
    out = []
    for item in items:
        if not isinstance(item, dict):
            continue
        out.append({"codigo": item.get("codigo"), "nome": item.get("nome")})
    return out


def normalize_movimentos(items: list[Any]) -> list[dict[str, Any]]:
    out = []
    for item in items:
        if not isinstance(item, dict):
            continue
        nome = str(item.get("nome") or "")
        data = item.get("dataHora")
        if nome:
            out.append({"codigo": item.get("codigo"), "nome": nome, "dataHora": data})
    out.sort(key=lambda x: str(x.get("dataHora") or ""), reverse=True)
    return out


def build_title(src: dict[str, Any], assuntos: list[dict[str, Any]]) -> str:
    tribunal = str(src.get("tribunal") or "").strip()
    classe = (src.get("classe") or {}).get("nome") or "Processo"
    assunto = next((str(a.get("nome") or "") for a in assuntos if a.get("nome")), "")
    parts = [str(classe)]
    if assunto:
        parts.append(assunto)
    if tribunal:
        parts.append(tribunal)
    return " - ".join(parts)[:180]


def build_similarity_text(
    src: dict[str, Any],
    assuntos: list[dict[str, Any]],
    movimentos: list[dict[str, Any]],
) -> str:
    classe = src.get("classe") or {}
    orgao = src.get("orgaoJulgador") or {}
    formato = src.get("formato") or {}
    sistema = src.get("sistema") or {}
    assunto_txt = "; ".join(
        str(a.get("nome") or "") for a in assuntos if a.get("nome")
    )
    movimentos_txt = "; ".join(
        str(m.get("nome") or "") for m in movimentos[:30] if m.get("nome")
    )
    ementa = str(src.get("ementa") or "").strip()
    parts = [
        f"Tribunal: {src.get('tribunal') or ''}",
        f"Grau: {src.get('grau') or ''}",
        f"Classe processual: {classe.get('nome') or ''}",
        f"Assuntos: {assunto_txt}",
        f"Orgao julgador: {orgao.get('nome') or ''}",
        f"Formato: {formato.get('nome') or ''}",
        f"Sistema: {sistema.get('nome') or ''}",
        f"Data de ajuizamento: {src.get('dataAjuizamento') or ''}",
        f"Movimentos processuais: {movimentos_txt}",
    ]
    if ementa:
        parts.append(f"Ementa: {ementa}")
    return "\n".join(p for p in parts if p.strip())


def infer_tipo(ementa: str, assuntos: list[dict[str, Any]], classe_nome: str) -> str:
    text = " ".join(
        [ementa, classe_nome, " ".join(str(a.get("nome") or "") for a in assuntos)]
    ).lower()
    if any(w in text for w in ["trabalhista", "clt", "horas extras", "fgts", "reclamante"]):
        return "Trabalhista"
    if any(w in text for w in ["consumidor", "cdc", "negativação", "plano de saúde", "fornecedor"]):
        return "Consumidor"
    if any(w in text for w in ["tributário", "tributario", "icms", "iss", "imposto", "fisco"]):
        return "Tributário"
    if any(w in text for w in ["previdenciário", "previdenciario", "inss", "aposentadoria", "benefício"]):
        return "Previdenciário"
    if any(w in text for w in ["criminal", "penal", "réu", "habeas corpus"]):
        return "Criminal"
    if any(w in text for w in ["família", "familia", "divórcio", "alimentos", "guarda", "inventário"]):
        return "Família"
    if any(w in text for w in ["software", "tecnologia", "licença de uso", "sistema"]):
        return "Tecnologia"
    return "Outros"


def infer_outcome(movimentos: list[dict[str, Any]]) -> str:
    for mov in movimentos:
        nome = str(mov.get("nome") or "").lower()
        for needle, label in OUTCOME_MAP.items():
            if needle in nome:
                return label
    return "desconhecido"


def dedupe_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    out = []
    for record in records:
        key = str(record.get("numeroProcesso") or record.get("id"))
        if key in seen:
            continue
        seen.add(key)
        out.append(record)
    return out


def add_embeddings(records: list[dict[str, Any]], model_path: str) -> None:
    if not records:
        return
    if not Path(model_path).exists():
        print(f"Modelo local nao encontrado; tentando baixar do HuggingFace: {model_path}")

    print(f"\nCarregando modelo de embeddings: {model_path}")
    model = SentenceTransformer(model_path)
    texts = [str(r.get("resumo") or "") for r in records]
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        with torch.inference_mode():
            embeddings = model.encode(
                batch,
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
        for record, embedding in zip(records[i : i + BATCH_SIZE], embeddings.tolist()):
            record["embedding"] = embedding
        print(f"  embeddings: {min(i + BATCH_SIZE, len(records))}/{len(records)}", end="\r")
    print()


def build_manifest(
    records: list[dict[str, Any]],
    tribunais: list[str],
    args: argparse.Namespace,
) -> dict[str, Any]:
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": "DataJud API Publica CNJ",
        "tribunais": tribunais,
        "max_por_tribunal": args.max,
        "total_records": len(records),
        "with_embeddings": not args.no_embeddings,
        "model_path": None if args.no_embeddings else args.model_path,
        "by_tribunal": dict(Counter(str(r.get("tribunal") or "") for r in records)),
        "by_tipo": dict(Counter(str(r.get("tipo") or "") for r in records)),
        "by_outcome": dict(Counter(str(r.get("outcome") or "") for r in records)),
    }


def maybe_download_colab(zip_path: Path, no_download: bool) -> None:
    if no_download:
        return
    try:
        from google.colab import files
    except Exception:
        return
    files.download(str(zip_path))


if __name__ == "__main__":
    main()
