"""Fixtures compartilhadas da suíte de testes."""

from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING

import pytest

from credit_sentinel.settings import Settings
from credit_sentinel.utils.seeds import make_rng

if TYPE_CHECKING:
    from pathlib import Path

    import numpy as np
    import pandas as pd

    from credit_sentinel.data.generator import ScenarioParams


@pytest.fixture
def settings_tmp(tmp_path: Path) -> Settings:
    """Settings isoladas, com todos os diretórios dentro de ``tmp_path``."""
    return Settings(_env_file=None, project_root=tmp_path)


@pytest.fixture
def rng() -> np.random.Generator:
    """Gerador aleatório determinístico (seed 42)."""
    return make_rng(42)


@pytest.fixture(scope="session")
def scenarios_path() -> Path:
    """Arquivo versionado de cenários do gerador."""
    from credit_sentinel.data.generator import SCENARIOS_FILE

    return Settings(_env_file=None).config_dir / SCENARIOS_FILE


@pytest.fixture(scope="session")
def reference_params(scenarios_path: Path) -> ScenarioParams:
    """Parâmetros do cenário ``reference``."""
    from credit_sentinel.data.generator import load_scenarios

    return load_scenarios(scenarios_path)["reference"]


@pytest.fixture(scope="session")
def df_reference_small(reference_params: ScenarioParams) -> pd.DataFrame:
    """Amostra pequena (500 linhas) da referência, seed 42."""
    from credit_sentinel.data.generator import generate_portfolio

    period = (date(2025, 7, 1), date(2026, 6, 30))
    return generate_portfolio(500, reference_params, make_rng(42), period)
