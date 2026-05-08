"""Utilidades compartilhadas entre modelos ML (device, caminhos)."""

from __future__ import annotations

from pathlib import Path

import torch


def get_torch_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_submodel_path(models_dir: str, name: str) -> Path:
    base = Path(models_dir).expanduser().resolve()
    direct = base / name
    if direct.is_dir():
        return direct
    parent = base.parent
    if (parent / name).is_dir():
        return parent / name
    return direct
