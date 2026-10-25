"""Fixtures compartilhadas da suíte de testes."""

from __future__ import annotations

import shutil
from datetime import date
from typing import TYPE_CHECKING

import pytest

from credit_sentinel.settings import Settings, get_settings
from credit_sentinel.utils.seeds import make_rng

if TYPE_CHECKING:
    from collections.abc import Iterator
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


@pytest.fixture
def project_tmp(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Settings]:
    """Projeto temporário com ``config/`` copiado, usado pelos testes da CLI."""
    shutil.copytree(Settings(_env_file=None).config_dir, tmp_path / "config")
    monkeypatch.setenv("CS_PROJECT_ROOT", str(tmp_path))
    get_settings.cache_clear()
    yield get_settings()
    get_settings.cache_clear()


PIPELINE_WINDOW_ROWS = 500


def _relax_slo_floor(config_dir: Path) -> None:
    """Baixa o piso de AUC do SLO4 no projeto de teste (baseline pequeno, AUC ~0,66)."""
    slo_file = config_dir / "slo_config.yaml"
    text = slo_file.read_text("utf-8")
    relaxed = text.replace("value: 0.75}", "value: 0.30}")
    assert relaxed != text
    slo_file.write_text(relaxed, "utf-8")


def _write_small_windows(settings: Settings) -> None:
    from credit_sentinel.drift.simulator import load_windows, simulate_window
    from credit_sentinel.utils.io import write_parquet

    for spec in load_windows(settings.config_dir / "simulation_scenarios.yaml"):
        small = simulate_window(
            spec.model_copy(update={"n_rows": PIPELINE_WINDOW_ROWS})
        )
        write_parquet(small, settings.production_dir / f"{spec.name}.parquet")


@pytest.fixture(scope="module")
def pipeline_project(
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[Settings]:
    """Projeto temporário com baseline treinado e janelas pequenas (uma vez por módulo).

    Contém referência de 1.500 linhas, o lote corrompido, o baseline e as três janelas
    de produção com 500 linhas; cópias intactas ficam em ``pristine/``. O piso do SLO4
    é rebaixado porque o baseline pequeno não alcança o AUC de 0,75 do projeto real.
    """
    from credit_sentinel.cli import main

    root = tmp_path_factory.mktemp("pipeline")
    shutil.copytree(Settings(_env_file=None).config_dir, root / "config")
    _relax_slo_floor(root / "config")
    patch = pytest.MonkeyPatch()
    patch.setenv("CS_PROJECT_ROOT", str(root))
    get_settings.cache_clear()
    assert main(["generate-data", "--rows", "1500"]) == 0
    assert main(["train", "--no-mlflow"]) == 0
    settings = get_settings()
    _write_small_windows(settings)
    shutil.copytree(settings.production_dir, root / "pristine")
    yield settings
    patch.undo()
    get_settings.cache_clear()


@pytest.fixture
def project_clean(pipeline_project: Settings) -> Settings:
    """Projeto do módulo com logs, relatórios e janelas no estado inicial."""
    shutil.rmtree(pipeline_project.logs_dir, ignore_errors=True)
    shutil.rmtree(pipeline_project.reports_dir, ignore_errors=True)
    shutil.rmtree(pipeline_project.production_dir)
    shutil.copytree(
        pipeline_project.project_root / "pristine", pipeline_project.production_dir
    )
    return pipeline_project
