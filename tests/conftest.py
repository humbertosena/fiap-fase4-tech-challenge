"""Fixtures compartilhadas da suíte de testes."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from credit_sentinel.settings import Settings
from credit_sentinel.utils.seeds import make_rng

if TYPE_CHECKING:
    from pathlib import Path

    import numpy as np


@pytest.fixture
def settings_tmp(tmp_path: Path) -> Settings:
    """Settings isoladas, com todos os diretórios dentro de ``tmp_path``."""
    return Settings(_env_file=None, project_root=tmp_path)


@pytest.fixture
def rng() -> np.random.Generator:
    """Gerador aleatório determinístico (seed 42)."""
    return make_rng(42)
