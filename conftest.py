"""Configuração global de pytest (env antes dos imports da API)."""

from __future__ import annotations

import os

os.environ.setdefault("JURISPRO_SKIP_PRELOAD", "1")
