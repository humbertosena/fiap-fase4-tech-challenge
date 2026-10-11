"""Valida o ambiente local antes do ``uv sync``.

Uso: ``python scripts/validate_env.py`` (exit 0 = pronto).
Usa apenas a biblioteca padrão, importando o módulo de checagem direto de ``src/``.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "credit_sentinel"))

from utils.env_check import run_checks, summarize

if __name__ == "__main__":
    sys.exit(summarize(run_checks()))
