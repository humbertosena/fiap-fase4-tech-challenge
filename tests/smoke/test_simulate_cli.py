"""Smoke tests do subcomando ``simulate``."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
import pytest

from credit_sentinel.cli import main
from credit_sentinel.drift.simulator import WINDOWS
from credit_sentinel.utils.io import file_sha256

if TYPE_CHECKING:
    from credit_sentinel.settings import Settings

pytestmark = pytest.mark.smoke


@pytest.fixture
def project(project_tmp: Settings) -> Settings:
    assert main(["generate-data", "--rows", "400"]) == 0
    return project_tmp


def test_simulate_writes_the_three_windows(
    project: Settings, capsys: pytest.CaptureFixture[str]
) -> None:
    capsys.readouterr()
    assert main(["simulate"]) == 0
    for name in WINDOWS:
        frame = pd.read_parquet(project.production_dir / f"{name}.parquet")
        assert frame.shape == (5000, 18)
    out = capsys.readouterr().out
    assert "mes_3_crise" in out
    assert "taxa base" in out
    assert "sha256" in out
    assert "renda_mensal" in out
    assert "desempregado" in out


def test_simulate_refuses_to_overwrite_without_force(project: Settings) -> None:
    assert main(["simulate"]) == 0
    assert main(["simulate"]) == 1
    assert project.production_dir.exists()


def test_force_regenerates_identical_files(project: Settings) -> None:
    path = project.production_dir / "mes_2.parquet"
    main(["simulate"])
    first = file_sha256(path)
    assert main(["simulate", "--force"]) == 0
    assert file_sha256(path) == first


def test_windows_option_selects_a_subset(project: Settings) -> None:
    assert main(["simulate", "--windows", "mes_2"]) == 0
    assert [p.name for p in project.production_dir.iterdir()] == ["mes_2.parquet"]


def test_unknown_window_is_an_error(
    project: Settings, capsys: pytest.CaptureFixture[str]
) -> None:
    capsys.readouterr()
    assert main(["simulate", "--windows", "mes_9"]) == 1
    assert "mes_9" in capsys.readouterr().err
    assert not project.production_dir.exists()


@pytest.mark.usefixtures("project_tmp")
def test_simulate_requires_the_reference(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["simulate"]) == 1
    assert "generate-data" in capsys.readouterr().out
